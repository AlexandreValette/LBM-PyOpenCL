# -*- coding: utf-8 -*-
"""OpenCL implementation of the D2Q9 / MRT lattice Boltzmann solver.

Three classes :
  - LBM      : physical parameters and initial state, computed on the CPU with NumPy
               (relaxation frequency, MRT matrix, equilibrium populations...).
  - LBM_OCL  : copies that state to an OpenCL device (GPU or CPU) and launches the kernels of
               kernels.cl. run(n) is the time loop ; everything heavy happens on the device.
  - DyeEmitter / RectDye : sources of a passive dye advected by the flow (visualisation only).

Data layout (the same everywhere, host and device)
  - a cell (ix, iy) of the nx * ny grid has the flat index  ix * ny + iy ;
  - the 9 populations f_0..f_8 are split in three arrays so that the device can read 4 of them
    at once as a float4 :   f0  (nx, ny)      the rest population
                            f14 (nx, ny, 4)   directions 1..4  (+x, +y, -x, -y)
                            f58 (nx, ny, 4)   directions 5..8  (diagonals)
  - `fin`  = populations after streaming (input of the collision),
    `fout` = populations after collision (input of the streaming), `feq` = equilibrium ;
  - `uvrho` (nx, ny, 4) holds (u, v, rho, unused) for each cell.

One time step, in LBM_OCL.run : bcs_sym -> inlet -> moments (= collision) -> streaming
-> optional dye advection. See DOCUMENTATION.md for the physics and the other files.
"""
import numpy as np
import pyopencl as cl
import ctypes as ct

class RectDye:
    """A rectangle of cells that emits dye : cells [cx_min, cx_max[ x [cy_min, cy_max[, with the given
    dye `value`, during `life_dt` time steps. With `randomize`, the value is random (0.75 .. 0.75 + value)."""
    def __init__(self,  cx_min,cx_max, cy_min, cy_max, value, life_dt, randomize = False):
        self.cx_min, self.cx_max, self.cy_min, self.cy_max, self.value, self.life_dt =  cx_min,cx_max, cy_min, cy_max, value, life_dt
        self.randomize=randomize
        
class DyeEmitter:
    """List of dye-emitting cells, ready to be copied to the device.

    `data` has one row per emitter : (flat cell index, unused, dye value, remaining life in time steps).
    The kernel update_dye_emitter writes the value in the dye field and decrements the life each step.
    The rows are used as a circular buffer of max_emitter entries."""
    max_emitter = 1025*3
    
    def __init__(self, shape2D):
        self.shape2D = shape2D
        self.ny = shape2D[1]
        self.shape   = (DyeEmitter.max_emitter,4)
        self.data  =np.zeros(self.shape, dtype=np.float32)
        self.pos = 0
        
    def add_rect_emitter(self, rectDye):
        """Append one emitter per cell of a RectDye."""

        if(rectDye.randomize):
            for j in range(rectDye.cx_min, rectDye.cx_max):
                for i in range(rectDye.cy_min, rectDye.cy_max):
                    self.data[self.pos,:] = np.array([j*self.ny+i, 0., np.random.rand(1)*rectDye.value+0.75, rectDye.life_dt], dtype=np.float32)
                    self.pos+=1
                    self.pos =  self.pos%DyeEmitter.max_emitter   
        else:
            for j in range(rectDye.cx_min, rectDye.cx_max):
                for i in range(rectDye.cy_min, rectDye.cy_max):
                    self.data[self.pos,:] = np.array([j*self.ny+i, 0., rectDye.value, rectDye.life_dt], dtype=np.float32)
                    self.pos+=1
                    self.pos =  self.pos%DyeEmitter.max_emitter    
                    
    def add_circular_emitter(self, cx, cy, r, value, life_dt, randomize = False):
        """Append the cells of a disc of radius r centred on (cx, cy) ; the value fades towards the border."""
        r2=r*r
        for i in range(-r,r):
            for j in range(-r,r):
                if((i**2+j**2)<r2): 
                    self.data[self.pos,:] = np.array([(cx+i)*self.ny+j+cy, 0., 2.-1.*(i**2+j**2)/r2, life_dt], dtype=np.float32)
                    self.pos+=1
                    self.pos =  self.pos%DyeEmitter.max_emitter    
                    
    def set_data(self, data):
        """Replace the whole emitter array (shape (max_emitter, 4), float32)."""
        self.data=data
        
    def get_data(self):
        """The emitter array, to copy to the device."""
        return self.data        

class LBM_OCL:
    """Runs an LBM on an OpenCL device.

    Usage : ocl_d = OCL_Driver(...) ; lbm = LBM(...) ; solver = LBM_OCL(ocl_d) ; solver.init_data(lbm)
            solver.update_obstacle(mask) ; solver.run(n)     (then get_rho_u_v() to read the result)

    The step kernels are launched on a 2D range of nx * ny work-items (one per cell), except the
    inlet kernel (one per row). To show the flow, add_postrun_texture_interop(image) registers a kernel
    that converts the velocity field into colors in an OpenCL image after each batch of steps.
    """
    def __init__(self, ocl_driver):
        """ocl_driver : an ocl_driver.OCL_Driver (context, queue and compiled kernels)."""
     
        self.ocl_d=ocl_driver
  
        self.post_lbm_functions = [] # List of function run after one lbm step
        self.post_run_functions = [] # List of function run after a btach of lbm step
        self.dye_emitters = None  
        self.interop = False
        
    def init_data(self, lbm):
        """Create all the device buffers from the initial state of an LBM object (host -> device copy)."""
        self.speedup=lbm.speedup
        self.lbm = lbm
        nx = lbm.nx ; ny = lbm.ny ; q= lbm.q
        
        self.shape3D = (9,nx,ny)
        self.shape2D = (nx,ny)
        self.shape2D_f4 = (nx,ny,4)
        self.size_qx = np.int32(nx*q)
        self.size_xy = np.int32(nx*ny)
        self.shape_inlet = lbm.fin0[0,:].shape

        self.vel_inlet = lbm.vel[:,0,:].flatten()        
        self.norm_uv = np.zeros((nx,ny), dtype = np.float32)

        mf = cl.mem_flags
        self.ocl_omega       = cl.Buffer(self.ocl_d.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(lbm.omega))
        self.ocl_noslip      = cl.Buffer(self.ocl_d.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lbm.noslip)
        self.ocl_obstacle    = cl.Buffer(self.ocl_d.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lbm.obstacle)
        self.ocl_c           = cl.Buffer(self.ocl_d.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lbm.c)
        self.ocl_t           = cl.Buffer(self.ocl_d.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lbm.t)        
        self.ocl_fin0         = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=lbm.fin0)
        self.ocl_fin14         = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=lbm.fin14)
        self.ocl_fin58         = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=lbm.fin58)        
        self.ocl_feq0         = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=lbm.feq0)
        self.ocl_feq14         = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=lbm.feq14)
        self.ocl_feq58         = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=lbm.feq58)        
        self.ocl_fout0        = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=lbm.fout0)
        self.ocl_fout14        = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=lbm.fout14)
        self.ocl_fout58        = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=lbm.fout58)
        self.ocl_M1S        = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=lbm.M1S)
        self.ocl_vel_inlet = cl.Buffer(self.ocl_d.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vel_inlet)

        self.ocl_uvrho          = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=lbm.uvrho)
        
        self.ocl_norm_uv = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=lbm.uvrho)
        self.texture_buffer = np.copy(lbm.uvrho)
        self.ocl_postbuffer_1= cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.norm_uv)        
        
        colormap_r = np.array([0.2298057, 0.6672529243333334, 0.968203399, 0.705673158], dtype=np.float32 )
        colormap_g = np.array([0.298717966,0.77917645699999, 0.720844, 0.01555616],      dtype=np.float32 )
        colormap_b = np.array([0.753683153, 0.99295, 0.61229299,0.1502328],              dtype=np.float32 )
        self.ocl_colormap_r  = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=colormap_r)
        self.ocl_colormap_g  = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=colormap_g)
        self.ocl_colormap_b  = cl.Buffer(self.ocl_d.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=colormap_b)
        self.set_texture_min_max()
        
        float_size = ct.sizeof(ct.c_float)
        char_size = ct.sizeof(ct.c_char)
        self.ocl_c_local         = cl.LocalMemory(char_size * int(lbm.c.size))
        self.ocl_t_local         = cl.LocalMemory(float_size * int(lbm.c.size/2))
        self.ocl_local_M1S       = cl.LocalMemory(float_size * int(lbm.M1S.flatten().size))

    
    def update_obstacle(self, obstacle):
        """Upload a boolean mask (nx, ny) of the solid cells."""
        cl.enqueue_copy(self.ocl_d.queue, self.ocl_obstacle, obstacle.flatten())
    
    def update_re(self, re):
        """Change the Reynolds number on the fly : recompute omega and the MRT matrix, and upload it."""
        self.lbm.update_re(re) 
        cl.enqueue_copy(self.ocl_d.queue, self.ocl_M1S,  self.lbm.M1S)
    
    def enable_dye(self, dye):
        """Activate the dye. dye : float32 array (nx, ny, 2), two copies of the field used alternately
        (one is read while the other is written, see read_dye_channel)."""
        self.ocl_dye = cl.Buffer(self.ocl_d.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=dye)
        self.read_dye_channel = np.uint8(0)
        self.post_run_functions.append(self._dye_to_texture)
        self.post_lbm_functions.append(self.dye)
    
    def add_dye_emitters(self, dye_emitters):
        """Upload a DyeEmitter (first call : create the buffer and register the emitter kernel ; later : refresh it)."""
        if(self.dye_emitters is None):
            self.dye_emitters = dye_emitters
            
            self.ocl_dye_emitters = cl.Buffer(self.ocl_d.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.dye_emitters.get_data())
            self.post_lbm_functions.append(self.update_dye_emitter)
        else:
            self.dye_emitters = dye_emitters
            #cl.enqueue_copy(self.queue, self.dye_emitters.data,  self.ocl_dye_emitters).wait()            
            cl.enqueue_copy(self.ocl_d.queue, self.ocl_dye_emitters, self.dye_emitters.get_data()).wait()
 
            
    def update_dye_emitter(self):
        """Kernel update_dye_emitter : write the active emitters in the dye field."""
        self.ocl_d.prg.update_dye_emitter(self.ocl_d.queue, self.dye_emitters.shape, None,self.ocl_dye_emitters,
                                         self.ocl_dye)
        
    def add_postrun_texture_interop(self, textureMem, value = "norm"):
        """Register the image (or OpenGL texture) in which the flow is drawn after each run().
        value : "norm" (velocity norm) or "curl" (curl of the velocity).
        The name comes from the OpenGL sharing mode ; with a plain cl.Image the interop is off."""
        self.interop = self.ocl_d.interop
        if(value=="norm"):
            self.post_run_functions.append(self._norm_uv_to_texture);
        elif(value=="curl"):
             self.post_run_functions.append(self._curl_to_texture);
 
        self._add_interop_texture(textureMem)
        
    def _add_interop_texture(self, textureMem):
        """Remember the target image and reset the color scale."""
        mf = cl.mem_flags
        self.textureMem = textureMem        
        self.set_texture_min_max()
        
    def set_texture_min_max(self, t_min=0., t_max=0.1):
        """Color scale of the display kernels. The values are multiplied by `speedup`, like the velocities."""
        minmax = np.array([t_min, t_max], dtype = np.float32)*self.speedup
        self.ocl_text_minmax   = cl.Buffer(self.ocl_d.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=minmax)
    
    def post_lbm(self):
        """Functions run after every time step (dye advection and emission)."""
        if (len(self.post_lbm_functions)!=0):
            for func in self.post_lbm_functions:
                func();
                
    def post_run(self):
        """Functions run once after a batch of steps (drawing into the image). Acquires / releases the
        image around them when it is shared with OpenGL."""
        if (len(self.post_run_functions)!=0):
            if(self.interop==True) : cl.enqueue_acquire_gl_objects(self.ocl_d.queue, [self.textureMem])   
            for func in self.post_run_functions:
                func();
            if(self.interop==True) : cl.enqueue_release_gl_objects(self.ocl_d.queue, [self.textureMem]) 
            
    def _dye_to_texture(self):
        """Put the dye in the alpha channel of the image."""
        # The dye goes in the alpha channel : read the texture, write a scratch image, copy it back
        # (no __read_write image, which is not available on every OpenCL device)
        if getattr(self, "dye_scratch", None) is None:
            fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
            self.dye_scratch = cl.create_image(self.ocl_d.ctx, cl.mem_flags.READ_WRITE, fmt, shape=self.shape2D)
        self.ocl_d.prg.dye_to_texture(self.ocl_d.queue, self.shape2D,self.ocl_d.local_size_2D , 
                                         self.ocl_dye, self.textureMem, self.dye_scratch, np.uint8(self.read_dye_channel) )
        self.ocl_d.prg.copy_texture(self.ocl_d.queue, self.shape2D,self.ocl_d.local_size_2D ,
                                         self.dye_scratch, self.textureMem)
        
    def _curl_to_texture(self):
        """Draw the curl of the velocity (signed, centred on 0) into the image."""
        self.ocl_d.prg.curl_to_texture(self.ocl_d.queue, self.shape2D,self.ocl_d.local_size_2D , self.textureMem,
                                         self.ocl_uvrho, 
                                         self.ocl_colormap_r, self.ocl_colormap_g, self.ocl_colormap_b, 
                                         self.ocl_text_minmax, self.ocl_obstacle) 
        
    def _norm_uv_to_texture(self):
        """Draw the norm of the velocity into the image (obstacles in green)."""
        self.ocl_d.prg.norm_uv_to_texture(self.ocl_d.queue, self.shape2D,self.ocl_d.local_size_2D , self.textureMem,
                                         self.ocl_uvrho,
                                         self.ocl_colormap_r, self.ocl_colormap_g, self.ocl_colormap_b, 
                                         self.ocl_text_minmax, self.ocl_obstacle)  
  
    def dye(self):
        """Advect the dye one step (semi-Lagrangian, kernel dye_bipolyinter), then swap the two dye copies."""
        self.ocl_d.prg.dye_bipolyinter(self.ocl_d.queue, self.shape2D,self.ocl_d.local_size_2D , self.ocl_dye,                                    
                                    self.ocl_uvrho,  self.ocl_obstacle, np.uint8(self.read_dye_channel))  
        self.read_dye_channel = (self.read_dye_channel+1)%2;            
        
    def norm_uv_to_buffer(self):
        """Same as _norm_uv_to_texture but into a plain buffer (see get_texture_from_buffer)."""
        self.ocl_d.prg.norm_u_to_buffer(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                         self.ocl_norm_uv, self.ocl_uvrho,
                                         self.ocl_colormap_r, self.ocl_colormap_g, self.ocl_colormap_b, 
                                         self.ocl_text_minmax)
        
    def curl_to_buffer(self):
        """Curl of the velocity into a plain buffer (see get_norm_uv_buffer)."""
        self.ocl_d.prg.curl_to_buffer(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                         self.ocl_norm_uv, self.ocl_uvrho)
        
    def bcs_sym(self):
        """Kernel bcs_sym : populations entering from the top / bottom (periodic) and the outlet (zero gradient)."""
        self.ocl_d.prg.bcs_sym(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                         self.ocl_fin0, self.ocl_fin14, self.ocl_fin58
                         ) 
        
    def collision(self):
        """BGK collision (single relaxation time, kernel `fout`). NOT used by run(), which uses the MRT
        collision of `moments` ; kept as a simpler alternative. Needs equilibrium() first."""
        self.ocl_d.prg.fout(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                        self.ocl_fout0, self.ocl_fout14,self.ocl_fout58,
                        self.ocl_fin0, self.ocl_fin14, self.ocl_fin58, 
                        self.ocl_feq0, self.ocl_feq14,self.ocl_feq58,  
                        self.ocl_obstacle,
                        self.ocl_noslip,
                        self.ocl_omega)
        
    def moments(self):
        """MRT collision (kernel `moments`) : computes rho and u, relaxes the moments, handles the bounce-back
        on obstacles. fin -> fout."""
        self.ocl_d.prg.moments(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                        self.ocl_fout0, self.ocl_fout14,self.ocl_fout58,
                        self.ocl_uvrho,
                        self.ocl_fin0, self.ocl_fin14, self.ocl_fin58, 
                        self.ocl_M1S, self.ocl_local_M1S,
                        self.ocl_obstacle)
        
    def inlet(self):
        """Kernel bcs_inlet : left boundary, imposed velocity. One work-item per row."""
        self.ocl_d.prg.bcs_inlet(self.ocl_d.queue, self.shape_inlet, self.ocl_d.local_size_inlet,
                        self.ocl_fin0,self.ocl_fin14,self.ocl_fin58,
                        self.ocl_uvrho, 
                        self.ocl_c, self.ocl_t,
                        self.ocl_vel_inlet,
                        self.size_xy)
        
    def equilibrium(self):
        """Equilibrium populations from uvrho (only needed by the BGK collision(), not by run())."""
        self.ocl_d.prg.equilibrium(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                            self.ocl_feq0, self.ocl_feq14, self.ocl_feq58, 
                            self.ocl_uvrho, 
                            self.ocl_c_local,  self.ocl_t_local ,
                            self.ocl_c,self.ocl_t)
    
    def macroscopic(self):
        """rho and u from fin (not used by run() : `moments` already does it)."""
        self.ocl_d.prg.rho_uv(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                        self.ocl_uvrho,  
                        self.ocl_fin0, self.ocl_fin14, self.ocl_fin58, 
                        self.ocl_c)      
        
    def streaming(self):
        """Kernel streaming : move every population to the neighbouring cell. fout -> fin."""
        self.ocl_d.prg.streaming(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                        self.ocl_fout0, self.ocl_fout14, self.ocl_fout58, 
                        self.ocl_fin0, self.ocl_fin14, self.ocl_fin58, 
                        self.ocl_c,
                        self.ocl_c_local )
        
    def get_rho_u_v(self):
        """Read back (u, v, rho, unused) for every cell : float32 array (nx, ny, 4)."""
        cl.enqueue_copy(self.ocl_d.queue, self.lbm.uvrho, self.ocl_uvrho).wait() 
        return np.copy(self.lbm.uvrho.reshape(self.shape2D_f4 ))
    
    def get_fout(self):
        """Read back the post-collision populations : (f0 (nx, ny), f14 (nx, ny, 4), f58 (nx, ny, 4))."""
        cl.enqueue_copy(self.ocl_d.queue, self.lbm.fout0, self.ocl_fout0).wait() 
        cl.enqueue_copy(self.ocl_d.queue, self.lbm.fout14, self.ocl_fout14).wait() 
        cl.enqueue_copy(self.ocl_d.queue, self.lbm.fout58, self.ocl_fout58).wait() 
        return np.copy(self.lbm.fout0.reshape(self.shape2D )), np.copy(self.lbm.fout14.reshape(self.shape2D_f4)),  np.copy(self.lbm.fout58.reshape(self.shape2D_f4 ))
    
    def get_norm_uv_buffer(self):
        """Read back the buffer filled by curl_to_buffer / norm_uv_to_buffer."""
        cl.enqueue_copy(self.ocl_d.queue, self.norm_uv, self.ocl_norm_uv).wait()    
        return self.norm_uv.reshape(self.shape2D)
    
    def get_texture_from_buffer(self):
        """RGBA colors of the velocity norm as a host array (nx, ny, 4), without any OpenGL texture."""
        self.norm_uv_to_buffer()
        cl.enqueue_copy(self.ocl_d.queue, self.texture_buffer, self.ocl_norm_uv).wait() 
        return self.texture_buffer.reshape(self.shape2D_f4 )    
   
    def run(self, nb_iter):
        """Advance the simulation by nb_iter time steps, then draw the registered images.
        Kernels are only queued : call queue.finish() (or read a result) to wait for them."""
        for i in range(nb_iter):            
            self.bcs_sym()             
            self.inlet()
            self.moments()
            self.streaming()              
            self.post_lbm()
        self.post_run()

class LBM:
    """Physical parameters and initial state of the simulation (host side, NumPy).

    Lattice units : one cell = one length unit, one time step = one time unit.
      uLB   = 0.04 * speedup    inlet velocity (must stay << 0.577 = speed of sound for the model to be valid)
      nulb  = uLB * l_ref / re  kinematic viscosity
      omega = 1 / (3 nulb + .5) relaxation frequency of the stress moments
    """
    def __init__(self, re, l_ref, nx, ny, speedup=1.):
        """re : Reynolds number ; l_ref : reference length in cells ; nx, ny : grid size ;
        speedup : multiplies uLB (faster flow per step, less accurate if too large)."""
        
        self.speedup = speedup
        self.re     = re     ; self.q = 9 ; self.l_ref=l_ref
        self.nx     = nx     ; self.ny     = ny        
        self.uLB    = 0.04*speedup        
        
        # Relaxation parameter
        self.nulb        = self.uLB*l_ref/re
        self.omega = 1.0 / (3.*self.nulb+0.5)
        
        # Lattice velocities c_i, in the order used everywhere : rest, +x, +y, -x, -y, then the four diagonals.
        self.c =np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype = np.int8)
     
        # Lattice weights.   
        self.t = np.array([4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,1./36.,1./36.,1./36.], dtype = np.float32)
        # Opposite direction of each population (bounce-back on obstacles)
        self.noslip = np.array([0,3,4,1,2,7,8,5,6]).astype(np.uint8)
      
        # Some constants : Speed of sound on the lattice, etc
        self.cs=1/np.sqrt(3);  self.cs2 = self.cs**2 ; self.cs22 = 2*self.cs2 ; self.cssq = 2.0/9.0                


        self.obstacle = np.zeros((nx, ny), dtype=bool)
        # Create vectors for computation
        self.vel = np.fromfunction(lambda d,x,y: (1-d)*self.uLB*(1.0+0.2*np.sin(y/4*2*np.pi)),(2,nx,ny), dtype=np.float32)
        self.vel = np.fromfunction(lambda d,x,y: (1-d)*self.uLB,(2,nx,ny), dtype=np.float32)
        self.vel[:,self.obstacle] = 0.
        #self.vel[:,1:,:] = 0.
        self.uvrho = np.zeros((self.nx, self.ny,4), dtype = np.float32) ;
        self.uvrho[:,:,2] = 1.
        self.uvrho[:,:,0] = self.vel[0,:,:]
        self.uvrho[:,:,1] = self.vel[1,:,:]
        
        self.feq0,  self.feq14,    self.feq58 = self.equilibrium(self.uvrho);         
        
        self.fin0 = self.feq0.copy() ; self.fin14 = self.feq14.copy() ; self.fin58 = self.feq58.copy() ; 
        self.fout0 = self.feq0.copy() ; self.fout14 = self.feq14.copy() ; self.fout58 = self.feq58.copy() ;      
   
        self.total_duration = 0.
        self.total_iter = 0.
        self.vel=self.vel.swapaxes(0,2)        
                
        # MRT : M maps the 9 populations to 9 moments, in this order :
        #   rho, e (energy), eps (energy square), jx, qx (energy flux x), jy, qy, pxx, pxy (stress)
        # The collision relaxes each moment towards its equilibrium at its own rate (matrix S below) :
        #   conserved moments (rho, jx, jy) : 0 ; e, eps : 1.4 ; qx, qy : 1.2 ; pxx, pxy : omega (sets the viscosity)
        # and f <- f - M^-1 S (m - m_eq) = f - M1S (m - m_eq), see the `moments` kernel.
        self.M = np.array([[1,1,1,1,1,1,1,1,1],
              [-4,-1,-1,-1,-1,2,2,2,2],
              [4,-2,-2,-2,-2,1,1,1,1],
              [0,1,0,-1,0,1,-1,-1,1],
              [0,-2,0,2,0,1,-1,-1,1],
              [0,0,1,0,-1,1,1,-1,-1],
              [0,0,-2,0,2,1,1,-1,-1],
              [0,1,-1,1,-1,0,0,0,0],
              [0,0,0,0,0,1,-1,1,-1]], dtype=np.float32)        
        S = np.identity(9, dtype=np.float32)*self.omega    
        S[0,0] = 0. ; S[1,1] = 1.4 ; S[2,2] = 1.4 ; S[3,3] = 0. ; S[4,4] = 1.2 ; S[5,5] = 0. ; S[6,6] = 1.2 ; 
        self.M1S = np.dot(np.linalg.inv(self.M),S).astype(np.float32)        
   
    def update_re(self, re):
        """Recompute the viscosity, omega and the MRT matrix M1S = M^-1 S for a new Reynolds number."""
        self.nulb        = self.uLB*self.l_ref/re
        self.omega = 1.0 / (3.*self.nulb+0.5)
        S = np.identity(9, dtype=np.float32)*self.omega    
        S[0,0] = 0. ; S[1,1] = 1.4 ; S[2,2] = 1.4 ; S[3,3] = 0. ; S[4,4] = 1.2 ; S[5,5] = 0. ; S[6,6] = 1.2 ; 
        self.M1S = np.dot(np.linalg.inv(self.M),S).astype(np.float32) 
    
    def equilibrium(self, uvrho):              # Equilibrium distribution function.
        """Equilibrium populations for the (u, v, rho) field uvrho (nx, ny, 4) : returns (feq0, feq14, feq58).
        f_i^eq = w_i rho (1 + 3 c_i.u + 4.5 (c_i.u)^2 - 1.5 u^2)."""
        u = np.zeros((2,self.nx, self.ny))
        u[0,:,:] = uvrho[:,:,0]
        u[1,:,:] = uvrho[:,:,1]
        cu   = 1./self.cs2 * np.dot(self.c,u.transpose(1,0,2))
        usqr = 1./(2*self.cs2)*(uvrho[:,:,0]**2+uvrho[:,:,1]**2)
        feq14 = np.zeros((self.nx,self.ny,4), dtype=np.float32)
        feq58 = np.zeros((self.nx,self.ny,4), dtype=np.float32)
        feq0 = np.zeros((self.nx,self.ny), dtype=np.float32)
        
        feq0 = uvrho[:,:,2]*self.t[0]*(1.+cu[0]+0.5*cu[0]**2-usqr)
        for i in range(1,5): 
            feq14[:,:,i-1] = uvrho[:,:,2]*self.t[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
        for i in range(5,9):     
            feq58[:,:,i-5] = uvrho[:,:,2]*self.t[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
        return feq0.astype(np.float32),  feq14.astype(np.float32),    feq58.astype(np.float32)