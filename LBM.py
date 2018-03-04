# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import sys
import numpy as np; from numpy.linalg import *
import matplotlib.pyplot as plt; from matplotlib import cm
import cProfile
import pstats
import time
import itertools
import pygame
import Obstacle_manager as Obstacle_manager
# OpenCL imports
from pyopencl.tools import get_gl_sharing_context_properties
import pyopencl as cl
import ctypes as ct
import matplotlib.path as mplPath
# OpenGL imports
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData

class RectDye:
    def __init__(self,  cx_min,cx_max, cy_min, cy_max, value, life_dt, randomize = False):
        self.cx_min, self.cx_max, self.cy_min, self.cy_max, self.value, self.life_dt =  cx_min,cx_max, cy_min, cy_max, value, life_dt
        self.randomize=randomize
        
class DyeEmitter:
    max_emitter = 1025*3
    
    def __init__(self, shape2D):
        self.shape2D = shape2D
        self.ny = shape2D[1]
        self.shape   = (DyeEmitter.max_emitter,4)
        self.data  =np.zeros(self.shape, dtype=np.float32)
        self.pos = 0
        
    def add_rect_emitter(self, rectDye):

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
        r2=r*r
        for i in range(-r,r):
            for j in range(-r,r):
                if((i**2+j**2)<r2): 
                    self.data[self.pos,:] = np.array([(cx+i)*self.ny+j+cy, 0., 2.-1.*(i**2+j**2)/r2, life_dt], dtype=np.float32)
                    self.pos+=1
                    self.pos =  self.pos%DyeEmitter.max_emitter    
                    
    def set_data(self, data):
        self.data=data
        
    def get_data(self):
        return self.data        

class LBM_OCL:
    def __init__(self, ocl_driver): 
     
        self.ocl_d=ocl_driver
  
        self.post_lbm_functions = [] # List of function run after one lbm step
        self.post_run_functions = [] # List of function run after a btach of lbm step
        self.dye_emitters = None  
        
    def init_data(self, lbm):
        ''
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
        cl.enqueue_copy(self.ocl_d.queue, self.ocl_obstacle, obstacle.flatten())
    
    def update_re(self, re):
        self.lbm.update_re(re) 
        cl.enqueue_copy(self.ocl_d.queue, self.ocl_M1S,  self.lbm.M1S)
    
    def enable_dye(self, dye):           
        self.ocl_dye = cl.Buffer(self.ocl_d.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=dye)
        self.read_dye_channel = np.uint8(0)
        self.post_run_functions.append(self._dye_to_texture)
        self.post_lbm_functions.append(self.dye)
    
    def add_dye_emitters(self, dye_emitters):
        if(self.dye_emitters is None):
            self.dye_emitters = dye_emitters
            
            self.ocl_dye_emitters = cl.Buffer(self.ocl_d.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.dye_emitters.get_data())
            self.post_lbm_functions.append(self.update_dye_emitter)
        else:
            self.dye_emitters = dye_emitters
            #cl.enqueue_copy(self.queue, self.dye_emitters.data,  self.ocl_dye_emitters).wait()            
            cl.enqueue_copy(self.ocl_d.queue, self.ocl_dye_emitters, self.dye_emitters.get_data()).wait()
 
            
    def update_dye_emitter(self):
        self.ocl_d.prg.update_dye_emitter(self.ocl_d.queue, self.dye_emitters.shape, None,self.ocl_dye_emitters,
                                         self.ocl_dye)
        
    def add_postrun_texture_interop(self, textureMem, value = "norm"):
        """ Supported values : norm, curl
        """
        self.interop=True
        if(value=="norm"):
            self.post_run_functions.append(self._norm_uv_to_texture);
        elif(value=="curl"):
             self.post_run_functions.append(self._curl_to_texture);
 
        self._add_interop_texture(textureMem)
        
    def _add_interop_texture(self, textureMem):
        mf = cl.mem_flags
        self.textureMem = textureMem        
        self.set_texture_min_max()
        
    def set_texture_min_max(self, t_min=0., t_max=0.1):
        minmax = np.array([t_min, t_max], dtype = np.float32)*self.speedup
        self.ocl_text_minmax   = cl.Buffer(self.ocl_d.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=minmax)
    
    def post_lbm(self):
        if (len(self.post_lbm_functions)!=0):
            for func in self.post_lbm_functions:
                func();
                
    def post_run(self):
        if (len(self.post_run_functions)!=0):
            if(self.interop==True) : cl.enqueue_acquire_gl_objects(self.ocl_d.queue, [self.textureMem])   
            for func in self.post_run_functions:
                func();
            if(self.interop==True) : cl.enqueue_release_gl_objects(self.ocl_d.queue, [self.textureMem]) 
            
    def _dye_to_texture(self):                 
        self.ocl_d.prg.dye_to_texture(self.ocl_d.queue, self.shape2D,self.ocl_d.local_size_2D , 
                                         self.ocl_dye, self.textureMem, np.uint8(self.read_dye_channel) )
        
    def _curl_to_texture(self):
        self.ocl_d.prg.curl_to_texture(self.ocl_d.queue, self.shape2D,self.ocl_d.local_size_2D , self.textureMem,
                                         self.ocl_uvrho, 
                                         self.ocl_colormap_r, self.ocl_colormap_g, self.ocl_colormap_b, 
                                         self.ocl_text_minmax, self.ocl_obstacle) 
        
    def _norm_uv_to_texture(self):  
        self.ocl_d.prg.norm_uv_to_texture(self.ocl_d.queue, self.shape2D,self.ocl_d.local_size_2D , self.textureMem,
                                         self.ocl_uvrho,
                                         self.ocl_colormap_r, self.ocl_colormap_g, self.ocl_colormap_b, 
                                         self.ocl_text_minmax, self.ocl_obstacle)  
  
    def dye(self):
        self.ocl_d.prg.dye_bipolyinter(self.ocl_d.queue, self.shape2D,self.ocl_d.local_size_2D , self.ocl_dye,                                    
                                    self.ocl_uvrho,  self.ocl_obstacle, np.uint8(self.read_dye_channel))  
        self.read_dye_channel = (self.read_dye_channel+1)%2;            
        
    def norm_uv_to_buffer(self):
        self.ocl_d.prg.norm_u_to_buffer(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                         self.ocl_norm_uv, self.ocl_uvrho,
                                         self.ocl_colormap_r, self.ocl_colormap_g, self.ocl_colormap_b, 
                                         self.ocl_text_minmax)
        
    def curl_to_buffer(self):
        self.ocl_d.prg.curl_to_buffer(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                         self.ocl_norm_uv, self.ocl_uvrho)
        
    def bcs_sym(self):
        self.ocl_d.prg.bcs_sym(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                         self.ocl_fin0, self.ocl_fin14, self.ocl_fin58
                         ) 
        
    def collision(self):
        self.ocl_d.prg.fout(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                        self.ocl_fout0, self.ocl_fout14,self.ocl_fout58,
                        self.ocl_fin0, self.ocl_fin14, self.ocl_fin58, 
                        self.ocl_feq0, self.ocl_feq14,self.ocl_feq58,  
                        self.ocl_obstacle,
                        self.ocl_noslip,
                        self.ocl_omega)
        
    def moments(self):
        self.ocl_d.prg.moments(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                        self.ocl_fout0, self.ocl_fout14,self.ocl_fout58,
                        self.ocl_uvrho,
                        self.ocl_fin0, self.ocl_fin14, self.ocl_fin58, 
                        self.ocl_M1S, self.ocl_local_M1S,
                        self.ocl_obstacle)
        
    def inlet(self):
        self.ocl_d.prg.bcs_inlet(self.ocl_d.queue, self.shape_inlet, self.ocl_d.local_size_inlet,
                        self.ocl_fin0,self.ocl_fin14,self.ocl_fin58,
                        self.ocl_uvrho, 
                        self.ocl_c, self.ocl_t,
                        self.ocl_vel_inlet,
                        self.size_xy)
        
    def equilibrium(self):
        self.ocl_d.prg.equilibrium(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                            self.ocl_feq0, self.ocl_feq14, self.ocl_feq58, 
                            self.ocl_uvrho, 
                            self.ocl_c_local,  self.ocl_t_local ,
                            self.ocl_c,self.ocl_t)
    
    def macroscopic(self):
        self.ocl_d.prg.rho_uv(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                        self.ocl_uvrho,  
                        self.ocl_fin0, self.ocl_fin14, self.ocl_fin58, 
                        self.ocl_c)      
        
    def streaming(self):
        self.ocl_d.prg.streaming(self.ocl_d.queue, self.shape2D, self.ocl_d.local_size_2D,
                        self.ocl_fout0, self.ocl_fout14, self.ocl_fout58, 
                        self.ocl_fin0, self.ocl_fin14, self.ocl_fin58, 
                        self.ocl_c,
                        self.ocl_c_local )
        
    def get_rho_u_v(self):
        cl.enqueue_copy(self.ocl_d.queue, self.lbm.uvrho, self.ocl_uvrho).wait() 
        return np.copy(self.lbm.uvrho.reshape(self.shape2D_f4 ))
    
    def get_fout(self):
        cl.enqueue_copy(self.ocl_d.queue, self.lbm.fout0, self.ocl_fout0).wait() 
        cl.enqueue_copy(self.ocl_d.queue, self.lbm.fout14, self.ocl_fout14).wait() 
        cl.enqueue_copy(self.ocl_d.queue, self.lbm.fout58, self.ocl_fout58).wait() 
        return np.copy(self.lbm.fout0.reshape(self.shape2D )), np.copy(self.lbm.fout14.reshape(self.shape2D_f4)),  np.copy(self.lbm.fout58.reshape(self.shape2D_f4 ))
    
    def get_norm_uv_buffer(self):
        cl.enqueue_copy(self.ocl_d.queue, self.norm_uv, self.ocl_norm_uv).wait()    
        return self.norm_uv.reshape(self.shape2D)
    
    def get_texture_from_buffer(self):
        self.norm_uv_to_buffer()
        cl.enqueue_copy(self.ocl_d.queue, self.texture_buffer, self.ocl_norm_uv).wait() 
        return self.texture_buffer.reshape(self.shape2D_f4 )    
   
    def run(self, nb_iter):
        for i in range(nb_iter):            
            self.bcs_sym()             
            self.inlet()
            self.moments()
            self.streaming()              
            self.post_lbm()
        self.post_run()

class LBM:
    def __init__(self, re, l_ref, nx, ny, speedup=1.):
        
        self.speedup = speedup
        self.re     = re     ; self.q = 9 ; self.l_ref=l_ref
        self.nx     = nx     ; self.ny     = ny        
        self.uLB    = 0.04*speedup        
        
        # Relaxation parameter
        self.nulb        = self.uLB*l_ref/re
        self.omega = 1.0 / (3.*self.nulb+0.5)
        
        # Lattice velocities.
        self.c =np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype = np.int8)
     
        # Lattice weights.   
        self.t = np.array([4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,1./36.,1./36.,1./36.], dtype = np.float32)
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
        self.nulb        = self.uLB*self.l_ref/re
        self.omega = 1.0 / (3.*self.nulb+0.5)
        S = np.identity(9, dtype=np.float32)*self.omega    
        S[0,0] = 0. ; S[1,1] = 1.4 ; S[2,2] = 1.4 ; S[3,3] = 0. ; S[4,4] = 1.2 ; S[5,5] = 0. ; S[6,6] = 1.2 ; 
        self.M1S = np.dot(np.linalg.inv(self.M),S).astype(np.float32) 
    
    def equilibrium(self, uvrho):              # Equilibrium distribution function.
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