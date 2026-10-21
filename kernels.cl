// OpenCL kernels of the lattice Boltzmann solver (D2Q9, MRT). Launched from lbm_opencl.py.
//
// Conventions used by every kernel
//  - One work-item per grid cell (ix, iy) = (get_global_id(0), get_global_id(1)) ; flat index = ix * ny + iy.
//  - Populations f_0..f_8 are split in f0 (float), f14 (float4 : f_1..f_4 in .x .y .z .w) and
//    f58 (float4 : f_5..f_8). Directions : 0 rest, 1 +x, 2 +y, 3 -x, 4 -y, 5 (+1,+1), 6 (-1,+1), 7 (-1,-1), 8 (+1,-1).
//  - uvrho is a float4 per cell : (u, v, rho, unused).
//  - `fin` = after streaming, `fout` = after collision.
//  - Time step : bcs_sym, bcs_inlet, moments (collision), streaming, then the optional dye / drawing kernels.
//  - Pointers carry no access qualifier : the OpenCL 2.0 / 3.0 compilers reject __read_only etc. on __global pointers.

// MRT collision. Reads fin ; computes rho, u, v (stored in uvrho) and the 9 moments ; relaxes the moments
// towards equilibrium with M1S = M^-1 S ; writes fout. On obstacle cells : bounce-back (opposite directions swapped).
__kernel void 
    moments(
        __global float *fout0, __global float4 *fout14, __global float4 *fout58,
        __global float4 *uvrho,
        __global float *fin0, __global float4 *fin14, __global float4 *fin58,
        __global float *M1S  , __local float *l_M1S,
        __global uchar *obstacle
        )
    {
      // Managing indexes
      const int ix     = get_global_id(0);   
      const int iy     = get_global_id(1);    
      const int ny = get_global_size(1);                
      const int two_d_index = ix*ny + iy;   
      
      // Read fin
      const float l_fin0 = fin0[two_d_index];
      const float4 l_fin14 = fin14[two_d_index];
      const float4 l_fin58 = fin58[two_d_index];
      
      // Let the first local node put the matrix in shared memory
      const int l1 = get_local_id(0);
      const int l2 = get_local_id(1); 
           
      barrier(CLK_LOCAL_MEM_FENCE);
      if ((l1 == 0 && l2==0)){ 
          for (uchar i =0;i<9*9;i++) l_M1S[i] = M1S[i];
      } 
      barrier(CLK_LOCAL_MEM_FENCE);
      
      if(obstacle[two_d_index]==0) {
          float rho;
          float4 l_moment14;
          float4 l_moment58;      
         
          // Compute rho = sum(fin)
          rho =               l_fin0 +     l_fin14.x +     l_fin14.y +     l_fin14.z +     l_fin14.w +     l_fin58.x +     l_fin58.y +     l_fin58.z +     l_fin58.w;
          // Compute other moments
          l_moment14.x = -4.f*l_fin0 -     l_fin14.x -     l_fin14.y -     l_fin14.z -     l_fin14.w + 2.f*l_fin58.x + 2.f*l_fin58.y + 2.f*l_fin58.z + 2.f*l_fin58.w;
          l_moment14.y =  4.f*l_fin0 - 2.f*l_fin14.x - 2.f*l_fin14.y - 2.f*l_fin14.z - 2.f*l_fin14.w +     l_fin58.x +     l_fin58.y +     l_fin58.z +     l_fin58.w;
          l_moment14.z =                   l_fin14.x                 -     l_fin14.z                 +     l_fin58.x -     l_fin58.y -     l_fin58.z +     l_fin58.w;
          l_moment14.w =             - 2.f*l_fin14.x                 + 2.f*l_fin14.z                 +     l_fin58.x -     l_fin58.y -     l_fin58.z +     l_fin58.w;      
          l_moment58.x  =                             +    l_fin14.y                 -     l_fin14.w +     l_fin58.x +     l_fin58.y -     l_fin58.z -     l_fin58.w;
          l_moment58.y =                              -2.f*l_fin14.y                 + 2.f*l_fin14.w +     l_fin58.x +     l_fin58.y -     l_fin58.z -     l_fin58.w;
          l_moment58.z =             +     l_fin14.x -     l_fin14.y +     l_fin14.z -     l_fin14.w;
          l_moment58.w =                                                                             +     l_fin58.x -     l_fin58.y +     l_fin58.z -     l_fin58.w;
    
          // Save rho u and v into global memory     
          uvrho[two_d_index] = (float4) (l_moment14.z/rho,l_moment58.x/rho,rho, 0.f);      
          
          // Compute m-meq      
          const float jx2 = l_moment14.z*l_moment14.z ;
          const float jy2 = l_moment58.x*l_moment58.x ;      
          const float jx2jy2 = 3*(jx2 +jy2);
        
          float delta_moment[9]; 
          delta_moment[0] = 0.f;            
          delta_moment[1] = l_moment14.x +2.f*rho -jx2jy2;
          delta_moment[2] = l_moment14.y - rho+jx2jy2;
          delta_moment[3] = 0.f;
          delta_moment[4] = l_moment14.w + l_moment14.z;
          delta_moment[5] = 0.f;
          delta_moment[6] = l_moment58.y + l_moment58.x;
          delta_moment[7] = l_moment58.z - jx2+jy2;
          delta_moment[8] = l_moment58.w - l_moment14.z*l_moment58.x;
          
          float l_fout[9]; 
          // Compute sumj matij*delta_momentj
          for (uchar i =0; i<9; i++)  {
                l_fout[i]=0.f;
                for (uchar j =0; j<9; j++) 
                  l_fout[i] += (l_M1S[j+9*i]*delta_moment[j]);
          }    
        
          fout0[two_d_index] = l_fin0-l_fout[0];
          fout14[two_d_index] = l_fin14-(float4) (l_fout[1], l_fout[2], l_fout[3], l_fout[4]);
          fout58[two_d_index] = l_fin58-(float4) (l_fout[5], l_fout[6], l_fout[7], l_fout[8]);
      
       } else {
          fout0[two_d_index] = fin0[two_d_index];           
          fout14[two_d_index] = (float4) (l_fin14.z, l_fin14.w, l_fin14.x, l_fin14.y);           
          fout58[two_d_index] = (float4) (l_fin58.z, l_fin58.w, l_fin58.x, l_fin58.y);           
      }
    }
    
// Regularized BGK collision (Latt & Chopard 2006), with an optional Smagorinsky local relaxation time (Hou et al. 1996).
// Alternative to `moments` : f_out = feq + (1 - omega) fneq_reg, where fneq_reg keeps only the part of f - feq carried
// by the stress tensor. use_smag != 0 : omega is replaced by a local value that grows with the strain (eddy viscosity).
// Same output as `moments` : fout, and (u, v, rho) in uvrho. Obstacles : bounce-back.
__kernel void
    moments_reg(
        __global float *fout0, __global float4 *fout14, __global float4 *fout58,
        __global float4 *uvrho,
        __global float *fin0, __global float4 *fin14, __global float4 *fin58,
        __global uchar *obstacle,
        float omega0, float cs_smag, int use_smag
        )
    {
      const int ix = get_global_id(0);
      const int iy = get_global_id(1);
      const int ny = get_global_size(1);
      const int idx = ix*ny + iy;

      const float l0 = fin0[idx];
      const float4 l14 = fin14[idx];
      const float4 l58 = fin58[idx];

      if (obstacle[idx] != 0) {
          fout0[idx] = l0;
          fout14[idx] = (float4) (l14.z, l14.w, l14.x, l14.y);
          fout58[idx] = (float4) (l58.z, l58.w, l58.x, l58.y);
          return;
      }

      const float f[9] = {l0, l14.x, l14.y, l14.z, l14.w, l58.x, l58.y, l58.z, l58.w};
      const float w[9]  = {4.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f};
      const float cx[9] = {0.f, 1.f, 0.f, -1.f, 0.f, 1.f, -1.f, -1.f, 1.f};
      const float cy[9] = {0.f, 0.f, 1.f, 0.f, -1.f, 1.f, 1.f, -1.f, -1.f};

      float rho = 0.f, jx = 0.f, jy = 0.f;
      for (int i = 0; i < 9; i++) { rho += f[i]; jx += cx[i]*f[i]; jy += cy[i]*f[i]; }
      const float ux = jx/rho;
      const float uy = jy/rho;
      uvrho[idx] = (float4) (ux, uy, rho, 0.f);
      const float usqr = 1.5f*(ux*ux + uy*uy);

      // Non-equilibrium part and its second moments (stress tensor)
      float feq[9], fneq[9];
      float pxx = 0.f, pyy = 0.f, pxy = 0.f;
      for (int i = 0; i < 9; i++) {
          const float cu = 3.f*(cx[i]*ux + cy[i]*uy);
          feq[i]  = w[i]*rho*(1.f + cu + 0.5f*cu*cu - usqr);
          fneq[i] = f[i] - feq[i];
          pxx += cx[i]*cx[i]*fneq[i];
          pyy += cy[i]*cy[i]*fneq[i];
          pxy += cx[i]*cy[i]*fneq[i];
      }

      float omega = omega0;
      if (use_smag != 0) {
          const float tau0 = 1.f/omega0;
          const float strain = sqrt(pxx*pxx + pyy*pyy + 2.f*pxy*pxy);
          omega = 2.f/(tau0 + sqrt(tau0*tau0 + 18.f*1.41421356f*cs_smag*cs_smag*strain/rho));
      }

      float o[9];
      for (int i = 0; i < 9; i++) {
          const float q = 4.5f*w[i]*((cx[i]*cx[i] - 1.f/3.f)*pxx + (cy[i]*cy[i] - 1.f/3.f)*pyy + 2.f*cx[i]*cy[i]*pxy);
          o[i] = feq[i] + (1.f - omega)*q;
      }
      fout0[idx]  = o[0];
      fout14[idx] = (float4) (o[1], o[2], o[3], o[4]);
      fout58[idx] = (float4) (o[5], o[6], o[7], o[8]);
    }

// BGK collision (single relaxation time omega). Alternative to `moments`, not used by the time loop.
__kernel void 
    fout(
        __global float *fout0, __global float4 *fout14, __global float4 *fout58,
        __global float *fin0, __global float4 *fin14, __global float4 *fin58,
        __global float *feq0, __global float4 *feq14, __global float4 *feq58,
        __global uchar *obstacle,
        __global char *noslip,
        __global float *omega
        )
    {
      // Managing indexes
      const int ix     = get_global_id(0);   
      const int iy     = get_global_id(1);    
      const int ny = get_global_size(1);                
      const int two_d_index = ix*ny + iy;       
     
      if(obstacle[two_d_index]==0) {
          const float temp = fin0[two_d_index];
          fout0[two_d_index]      = temp - omega[0]*(temp - feq0[two_d_index]);
                   
          float4 temp4 = fin14[two_d_index];
          fout14[two_d_index]      = temp4 - omega[0]*(temp4 - feq14[two_d_index]); 
                  
          temp4 = fin58[two_d_index];
          fout58[two_d_index]      = temp4 - omega[0]*(temp4 - feq58[two_d_index]);         
      }
      else {
          fout0[two_d_index] = fin0[two_d_index];           
          fout14[two_d_index] = (float4) (fin14[two_d_index].z, fin14[two_d_index].w, fin14[two_d_index].x, fin14[two_d_index].y);           
          fout58[two_d_index] = (float4) (fin58[two_d_index].z, fin58[two_d_index].w, fin58[two_d_index].x, fin58[two_d_index].y);           
      }
    }



// Streaming : fin(x, y)[i] = fout(x - c_i)[i]. Populations that would come from outside the domain are set to 0
// (they are filled afterwards by bcs_sym / bcs_inlet).
__kernel void 
    streaming(
        __global float *fout0, __global float4 *fout14, __global float4 *fout58,
        __global float *fin0, __global float4 *fin14, __global float4 *fin58,
        __global char2 *c,
        __local char2 *local_c
        )
    {
 
      // Managing indexes
      const int ix     = get_global_id(0);   
      const int iy     = get_global_id(1); 
      const int nx = get_global_size(0);    
      const int ny = get_global_size(1);                
      const int two_d_index = ix*ny + iy;       
      
      fin0[two_d_index] = fout0[two_d_index];
      const float4 l_fout14 = fout14[two_d_index];
      const float4 l_fout58 = fout58[two_d_index];      

      if(ix<nx-1 && ix>0 && iy<ny-1 && iy>0) {
          fin14[two_d_index] = (float4) (fout14[two_d_index-ny].x, fout14[two_d_index-1].y, fout14[two_d_index+ny].z, fout14[two_d_index+1].w)    ;        
          fin58[two_d_index] = (float4) (fout58[two_d_index-ny-1].x, fout58[two_d_index+ny-1].y, fout58[two_d_index+ny+1].z, fout58[two_d_index-ny+1].w);
      }
      else {
                
          if(ix==nx-1 && iy==ny-1) {
              fin14[two_d_index] = (float4) (fout14[two_d_index-ny].x, fout14[two_d_index-1].y,0.f, 0.f)    ;        
              fin58[two_d_index] = (float4) (fout58[two_d_index-ny-1].x, 0.f, 0.f, 0.f);
              return;          
          }          
          if(ix==0 && iy==ny-1) {
              fin14[two_d_index] = (float4) (0.f, fout14[two_d_index-1].y, fout14[two_d_index+ny].z, 0.f)    ;        
              fin58[two_d_index] = (float4) (0.f, fout58[two_d_index+ny-1].y, 0.f, 0.f);
              return;      
          }          
          if(ix==0 && iy==0) {
              fin14[two_d_index] = (float4) (0.f, 0.f, fout14[two_d_index+ny].z, fout14[two_d_index+1].w)    ;        
              fin58[two_d_index] = (float4) (0.f, 0.f, fout58[two_d_index+ny+1].z, 0.f);
              return;          
          }          
          if(ix==nx-1 && iy==0) {
              fin14[two_d_index] = (float4) (fout14[two_d_index-ny].x, 0.f, 0.f, fout14[two_d_index+1].w)    ;        
              fin58[two_d_index] = (float4) (0.f, 0.f, 0.f, fout58[two_d_index-ny+1].w);
              return;      
          } 
          if(ix==nx-1) {
              fin14[two_d_index] = (float4) (fout14[two_d_index-ny].x, fout14[two_d_index-1].y, 0.f, fout14[two_d_index+1].w)    ;        
              fin58[two_d_index] = (float4) (fout58[two_d_index-ny-1].x, 0.f, 0.f, fout58[two_d_index-ny+1].w);
              return;
          }  
          if(iy==ny-1) {
              fin14[two_d_index] = (float4) (fout14[two_d_index-ny].x, fout14[two_d_index-1].y, fout14[two_d_index+ny].z, 0.f)    ;        
              fin58[two_d_index] = (float4) (fout58[two_d_index-ny-1].x, fout58[two_d_index+ny-1].y, 0.f, 0.f);
              return;
          }          
          if(ix==0)  {
              fin14[two_d_index] = (float4) (0.f, fout14[two_d_index-1].y, fout14[two_d_index+ny].z, fout14[two_d_index+1].w)    ;        
              fin58[two_d_index] = (float4) (0.f, fout58[two_d_index+ny-1].y, fout58[two_d_index+ny+1].z, 0.f);
              return;
          }          
          if(iy==0)   {
              fin14[two_d_index] = (float4) (fout14[two_d_index-ny].x, 0.f, fout14[two_d_index+ny].z, fout14[two_d_index+1].w)    ;        
              fin58[two_d_index] = (float4) (0.f, 0.f, fout58[two_d_index+ny+1].z, fout58[two_d_index-ny+1].w);
              return;    
          } 
      } 
    } 


// Equilibrium populations from (u, v, rho). Only used by the BGK collision.
__kernel void 
    equilibrium(
         __global float *feq0, __global float4 *feq14, __global float4 *feq58,
        __global float4 *uvrho,
        __local char2 *local_c, __local float *local_t,
        __global char2 *c,
        __global float *t
        )
    {

      // Managing indexes
      const int ix     = get_global_id(0);   
      const int iy     = get_global_id(1); 
      const int ny = get_global_size(1);                
      const int two_d_index = ix*ny + iy;   
   
      const float4 m_local_uvrho = uvrho[two_d_index];
      float cur_c_dot_u;
      const float usqr= (m_local_uvrho.x*m_local_uvrho.x + m_local_uvrho.y*m_local_uvrho.y)*1.5;
      
      const float r49 = 4.f/9.f;
      const float r19 = 1.f/9.f;
      const float r136 = 1.f/36.f;
      
      //printf("%d %d %f %f %f\n", ix,iy, r49, m_local_uvrho.z, usqr);
      feq0[two_d_index] = r49*m_local_uvrho.z*(1.-usqr);      
 
      cur_c_dot_u = (m_local_uvrho.x)*3.; 
      feq14[two_d_index].x = r19*m_local_uvrho.z*(1.+cur_c_dot_u+0.5*cur_c_dot_u*cur_c_dot_u-usqr); 

      cur_c_dot_u = (m_local_uvrho.y)*3.; 
      feq14[two_d_index].y = r19*m_local_uvrho.z*(1.+cur_c_dot_u+0.5*cur_c_dot_u*cur_c_dot_u-usqr); 

      cur_c_dot_u = (-m_local_uvrho.x)*3.; 
      feq14[two_d_index].z = r19*m_local_uvrho.z*(1.+cur_c_dot_u+0.5*cur_c_dot_u*cur_c_dot_u-usqr); 

      cur_c_dot_u = (-m_local_uvrho.y)*3.; 
      feq14[two_d_index].w = r19*m_local_uvrho.z*(1.+cur_c_dot_u+0.5*cur_c_dot_u*cur_c_dot_u-usqr); 

      cur_c_dot_u = (m_local_uvrho.x +m_local_uvrho.y)*3.; 
      feq58[two_d_index].x = r136*m_local_uvrho.z*(1.+cur_c_dot_u+0.5*cur_c_dot_u*cur_c_dot_u-usqr); 


      cur_c_dot_u = (-m_local_uvrho.x + m_local_uvrho.y)*3.; 
      feq58[two_d_index].y = r136*m_local_uvrho.z*(1.+cur_c_dot_u+0.5*cur_c_dot_u*cur_c_dot_u-usqr); 
      
      cur_c_dot_u = (-m_local_uvrho.x -m_local_uvrho.y)*3.; 
      feq58[two_d_index].z = r136*m_local_uvrho.z*(1.+cur_c_dot_u+0.5*cur_c_dot_u*cur_c_dot_u-usqr); 

      cur_c_dot_u = (m_local_uvrho.x - m_local_uvrho.y)*3.; 
      feq58[two_d_index].w = r136*m_local_uvrho.z*(1.+cur_c_dot_u+0.5*cur_c_dot_u*cur_c_dot_u-usqr); 
    }    
   


// rho and u, v from fin. Not used by the time loop (`moments` computes them).
__kernel void 
    rho_uv(
        __global float4 *uvrho,
         __global float *fin0, __global float4 *fin14, __global float4 *fin58,
        __global char2 *c
        )
    {
 
      // Managing indexes
      const int ix     = get_global_id(0);   
      const int iy     = get_global_id(1); 
      const int nx = get_global_size(0);    
      const int ny = get_global_size(1);                
      const int two_d_index = ix*ny + iy;   
      
      float4 new_uvrho;
      
      // Compute 
	  
	   new_uvrho.z = fin0[two_d_index] + fin14[two_d_index].x + fin14[two_d_index].y + fin14[two_d_index].z + fin14[two_d_index].w;
	   new_uvrho.z +=fin58[two_d_index].x + fin58[two_d_index].y + fin58[two_d_index].z+ + fin58[two_d_index].w;
     

      new_uvrho.x = fin14[two_d_index].x -fin14[two_d_index].z + fin58[two_d_index].x - fin58[two_d_index].y - fin58[two_d_index].z+  fin58[two_d_index].w;
      new_uvrho.y = fin14[two_d_index].y  - fin14[two_d_index].w +fin58[two_d_index].x + fin58[two_d_index].y -fin58[two_d_index].z- fin58[two_d_index].w;
      
      new_uvrho.x = new_uvrho.x /new_uvrho.z;
      new_uvrho.y = new_uvrho.y /new_uvrho.z;
      
      uvrho[two_d_index] = new_uvrho;

    }




// Boundaries : top and bottom walls take the populations coming from the opposite side (periodic),
// the outlet (last column) copies the previous column (zero gradient).
__kernel void 
    bcs_sym(
        __global float *fin0, __global float4 *fin14, __global float4 *fin58
        )
    {
	
      // Managing indexes
      const int ix     = get_global_id(0);   
      const int iy     = get_global_id(1); 
      const int nx = get_global_size(0);    
      const int ny = get_global_size(1);                
      const int two_d_index = ix*ny + iy;   
      
      // Top wall
      if(iy==(ny-1)) {
          //4
          fin14[two_d_index].w = fin14[two_d_index-(ny-1)].w;
          //7
          //8
          fin58[two_d_index].z = fin58[two_d_index-(ny-1)].z;
          fin58[two_d_index].w = fin58[two_d_index-(ny-1)].w;
      }
      // Bottom wall
      if(iy==0) {
          //2
          fin14[two_d_index].y = fin14[two_d_index + ny -1].y;
          // 5 et 6
          fin58[two_d_index].x = fin58[two_d_index + ny -1].x;
          fin58[two_d_index].y = fin58[two_d_index + ny -1].y;
      }            
                        
      // Outlet
      if(ix==(nx-1)) {
          //3 
          fin14[two_d_index].z = fin14[two_d_index - ny].z;
          //6 7 
          fin58[two_d_index].y = fin58[two_d_index - ny].y;
          fin58[two_d_index].z = fin58[two_d_index - ny].z;
      }          
	

    }


// Inlet (column ix = 0), one work-item per row : velocity imposed, density deduced from the known populations,
// the unknown populations 1, 5, 8 are set to their equilibrium.
__kernel void 
    bcs_inlet(
        __global float *fin0, __global float4 *fin14, __global float4 *fin58,
        __global float4 *uvrho,
        __global char2 *c,
        __global float *t,
        __global float2 *vel_inlet,
        int size_qx        
        )
    {
	   
      const int iy     = get_global_id(0);   
      const int index_two_d = iy; 
    
      const float2 local_uv = vel_inlet[iy];      
      const float cst1 = 1./(1.-local_uv.x);
      
      const float4 l_fin14 = fin14[index_two_d];
      const float4 l_fin58 = fin58[index_two_d];
      
      const float local_rho = 2.*cst1*(l_fin14.z + l_fin58.z + l_fin58.y)
                          +cst1*(fin0[index_two_d] + l_fin14.w + l_fin14.y)  ;          
      
      uvrho[index_two_d]   =  (float4) (local_uv.x, local_uv.y, local_rho,0.f) ;
      
      const float usqr        = (local_uv.x*local_uv.x + local_uv.y*local_uv.y)*1.5;
      float cur_c_dot_u;
      
      uchar iq=1;
      //printf("%v2d %f\n",c[iq],t[iq] );
      cur_c_dot_u = (c[iq].x*local_uv.x + c[iq].y*local_uv.y)*3.;
      fin14[index_two_d].x = t[iq]*local_rho*(1.+cur_c_dot_u+0.5*cur_c_dot_u*cur_c_dot_u-usqr);
      
      iq=8;
      cur_c_dot_u = (c[iq].x*local_uv.x + c[iq].y*local_uv.y)*3.;
      fin58[index_two_d].w = t[iq]*local_rho*(1.+cur_c_dot_u+0.5*cur_c_dot_u*cur_c_dot_u-usqr);
      
      iq=5;
      cur_c_dot_u = (c[iq].x*local_uv.x + c[iq].y*local_uv.y)*3.;
      fin58[index_two_d].x = t[iq]*local_rho*(1.+cur_c_dot_u+0.5*cur_c_dot_u*cur_c_dot_u-usqr);
	  
    }

// Inlet of Zou & He (column ix = 0), one work-item per row : same density as bcs_inlet, but populations 1, 5, 8 are
// obtained by bounce-back of the non-equilibrium part, which gives the right momentum and shear stress.
__kernel void
    bcs_inlet_zouhe(
        __global float *fin0, __global float4 *fin14, __global float4 *fin58,
        __global float4 *uvrho,
        __global float2 *vel_inlet
        )
    {
      const int iy = get_global_id(0);
      const float2 uv = vel_inlet[iy];
      const float4 l14 = fin14[iy];
      const float4 l58 = fin58[iy];
      const float rho = (fin0[iy] + l14.y + l14.w + 2.f*(l14.z + l58.z + l58.y))/(1.f - uv.x);
      uvrho[iy] = (float4) (uv.x, uv.y, rho, 0.f);
      fin14[iy].x = l14.z + 2.f/3.f*rho*uv.x;
      fin58[iy].x = l58.z - 0.5f*(l14.y - l14.w) + rho*uv.x/6.f + 0.5f*rho*uv.y;
      fin58[iy].w = l58.y + 0.5f*(l14.y - l14.w) + rho*uv.x/6.f - 0.5f*rho*uv.y;
    }

// Convective outlet (last column), one work-item per row, to run after bcs_sym : the incoming populations 3, 6, 7 are
// advected at speed uc : f(outlet, t+1) = (f(outlet, t) + uc f(outlet-1, t+1)) / (1 + uc). `prev` (3 rows of ny floats)
// keeps f(outlet, t) from one step to the next.
__kernel void
    bcs_outlet_convective(
        __global float4 *fin14, __global float4 *fin58,
        __global float *prev,
        int nx, float uc
        )
    {
      const int iy = get_global_id(0);
      const int ny = get_global_size(0);
      const int last = (nx-1)*ny + iy;
      const int before = (nx-2)*ny + iy;
      const float n3 = (prev[iy]        + uc*fin14[before].z)/(1.f + uc);
      const float n6 = (prev[ny + iy]   + uc*fin58[before].y)/(1.f + uc);
      const float n7 = (prev[2*ny + iy] + uc*fin58[before].z)/(1.f + uc);
      fin14[last].z = n3;
      fin58[last].y = n6;
      fin58[last].z = n7;
      prev[iy] = n3;
      prev[ny + iy] = n6;
      prev[2*ny + iy] = n7;
    }

// Colors of the velocity norm written into a plain float4 buffer (blue -> red color map).
__kernel void 
    norm_u_to_buffer(
        __global float4 *norm_uv,
        __global float4 *uvrho,
        __global float4 *colormap_r,
        __global float4 *colormap_g,
        __global float4 *colormap_b,
        __global float2 *norm_minmax
         )
    {      
      int2 pixelcoord = (int2) (get_global_id(0), get_global_id(1));      
      const int two_d_index = get_global_id(0)*get_global_size(1) + get_global_id(1); 
      
      const float local_u = uvrho[two_d_index].x;
      const float local_v = uvrho[two_d_index].y;
      
      // Compute norm
      float norm = sqrt(local_u*local_u+local_v*local_v);
      float4  val;
      
      // Manage min/max
      if(norm<=norm_minmax[0].x) {
              val = (float4) (colormap_r[0].x,colormap_g[0].x,colormap_b[0].x,1.);
              norm_uv[two_d_index] =  val;
              return;                
      }
      if(norm>=norm_minmax[0].y) {
              val = (float4) (colormap_r[0].w,colormap_g[0].w,colormap_b[0].w,1.);
              norm_uv[two_d_index] =  val;
              return;                
      }
      
      // Other size interpolation
      norm = (norm-norm_minmax[0].x)/(norm_minmax[0].y-norm_minmax[0].x);
      
      if(norm<0.33) {
              const float norm1 = 3*norm;
              val.x =   colormap_r[0].x + norm1*(colormap_r[0].y-colormap_r[0].x);
              val.y =   colormap_g[0].x + norm1*(colormap_g[0].y-colormap_g[0].x);
              val.z =   colormap_b[0].x + norm1*(colormap_b[0].y-colormap_b[0].x);
              val.w =   1.f;            
      }
      else if (norm<0.66) {
              const float norm2 = 3*norm-1;
              val.x =   colormap_r[0].y + norm2*(colormap_r[0].z-colormap_r[0].y);
              val.y =   colormap_g[0].y + norm2*(colormap_g[0].z-colormap_g[0].y);
              val.z =   colormap_b[0].y + norm2*(colormap_b[0].z-colormap_b[0].y);
              val.w =   1.f;    
      }
      else {
              const float norm3 = 3*norm-2;
              val.x =   colormap_r[0].z + norm3*(colormap_r[0].w-colormap_r[0].z);
              val.y =   colormap_g[0].z + norm3*(colormap_g[0].w-colormap_g[0].z);
              val.z =   colormap_b[0].z + norm3*(colormap_b[0].w-colormap_b[0].z);
              val.w =   1.f;    
      }      
 
      norm_uv[two_d_index] =  val;      
    }
          
    
// Colors of the velocity norm written into an image. Obstacles are green.
// The scale is norm_minmax = (min, max) : norm <= min gives the first color, >= max the last one.
__kernel void 
    norm_uv_to_texture(
        __write_only image2d_t texture_normuv,
        __global float4 *uvrho,
        __global float4 *colormap_r,
        __global float4 *colormap_g,
        __global float4 *colormap_b,
        __global float2 *norm_minmax,
        __global uchar *obstacle
         )
    {
 
      int2 pixelcoord = (int2) (get_global_id(0), get_global_id(1));      
      const int two_d_index = get_global_id(0)*get_global_size(1) + get_global_id(1); 
      
      const float local_u = uvrho[two_d_index].x;
      const float local_v = uvrho[two_d_index].y;
      
      if(obstacle[two_d_index]!=0){
          write_imagef(texture_normuv, pixelcoord, (float4) (0.,0.5,0.,1.));
          return;
      }
      
      // Compute norm
      float norm = sqrt(local_u*local_u+local_v*local_v);
      float4  val;
      
      // Manage min/max
      if(norm<=norm_minmax[0].x) {
              val = (float4) (colormap_r[0].x,colormap_g[0].x,colormap_b[0].x,1.);
              write_imagef(texture_normuv, pixelcoord, val);
              return;                
      }
      if(norm>=norm_minmax[0].y) {
              val = (float4) (colormap_r[0].w,colormap_g[0].w,colormap_b[0].w,1.);
              write_imagef(texture_normuv, pixelcoord, val);
              return;                
      }
      
      // Other size interpolation
      norm = (norm-norm_minmax[0].x)/(norm_minmax[0].y-norm_minmax[0].x);
      
      if(norm<0.33) {
              const float norm1 = 3*norm;
              val.x =   colormap_r[0].x + norm1*(colormap_r[0].y-colormap_r[0].x);
              val.y =   colormap_g[0].x + norm1*(colormap_g[0].y-colormap_g[0].x);
              val.z =   colormap_b[0].x + norm1*(colormap_b[0].y-colormap_b[0].x);
              val.w =   1.f;            
      }
      else if (norm<0.66) {
              const float norm2 = 3*norm-1;
              val.x =   colormap_r[0].y + norm2*(colormap_r[0].z-colormap_r[0].y);
              val.y =   colormap_g[0].y + norm2*(colormap_g[0].z-colormap_g[0].y);
              val.z =   colormap_b[0].y + norm2*(colormap_b[0].z-colormap_b[0].y);
              val.w =   1.f;    
      }
      else {
              const float norm3 = 3*norm-2;
              val.x =   colormap_r[0].z + norm3*(colormap_r[0].w-colormap_r[0].z);
              val.y =   colormap_g[0].z + norm3*(colormap_g[0].w-colormap_g[0].z);
              val.z =   colormap_b[0].z + norm3*(colormap_b[0].w-colormap_b[0].z);
              val.w =   1.f;    
      }     
      
       
      // Apply colormap to texture_normuv
      write_imagef(texture_normuv, pixelcoord, val);
      
    }
    
// Curl of the velocity (dv/dx - du/dy by finite differences) into a float buffer.
__kernel void 
    curl_to_buffer(
    __global float *norm_uv,
        __global float4 *uvrho

         )
    {
      // Managing indexes
      const int size_0 = get_global_size(0);
      const int size_1 = get_global_size(1);
      const int two_d_index = get_global_id(0)*size_1 + get_global_id(1); 
      
      const float local_u = uvrho[two_d_index].x;
      const float local_v = uvrho[two_d_index].y;
      
      const float local_u_y1 = uvrho[two_d_index+1].x;
      const float local_v_x1 = uvrho[two_d_index+size_1].y;
      
      // Compute norm
      if(get_global_id(1)<size_1-2 && get_global_id(0)<size_0-2)
              norm_uv[two_d_index] = (local_v_x1-local_v)-(local_u_y1-local_u);
       
      
    }
    
// Colors of the signed curl into an image : -max -> first color, 0 -> middle, +max -> last color.
__kernel void 
    curl_to_texture(
        __write_only image2d_t texture_normuv,
        __global float4 *uvrho,
        __global float4 *colormap_r,
        __global float4 *colormap_g,
        __global float4 *colormap_b,
        __global float2 *norm_minmax,
        __global uchar *obstacle
         )
    {
 
      int2 pixelcoord = (int2) (get_global_id(0), get_global_id(1));  
      const int size_0 = get_global_size(0);
      const int size_1 = get_global_size(1);
      
      const int two_d_index = get_global_id(0)*size_1 + get_global_id(1); 
      
      const float local_u = uvrho[two_d_index].x;
      const float local_v = uvrho[two_d_index].y;
      
      if(obstacle[two_d_index]!=0){
          write_imagef(texture_normuv, pixelcoord, (float4) (0.,0.5,0.,1.));
          return;
      }

      // Curl by finite differences (zero on the last row/column)
      float curl = 0.f;
      if(get_global_id(1)<size_1-1 && get_global_id(0)<size_0-1) {
          const float local_u_y1 = uvrho[two_d_index+1].x;
          const float local_v_x1 = uvrho[two_d_index+size_1].y;
          curl = (local_v_x1-local_v)-(local_u_y1-local_u);
      }

      // Signed curl, symmetric around 0 : curl = -max -> first color, +max -> last color
      float norm = clamp(0.5f + 0.5f*curl/norm_minmax[0].y, 0.f, 1.f);
      float4 val;
      if(norm<0.33f) {
              const float n1 = 3*norm;
              val = (float4) (colormap_r[0].x + n1*(colormap_r[0].y-colormap_r[0].x),
                              colormap_g[0].x + n1*(colormap_g[0].y-colormap_g[0].x),
                              colormap_b[0].x + n1*(colormap_b[0].y-colormap_b[0].x), 1.f);
      }
      else if (norm<0.66f) {
              const float n2 = 3*norm-1;
              val = (float4) (colormap_r[0].y + n2*(colormap_r[0].z-colormap_r[0].y),
                              colormap_g[0].y + n2*(colormap_g[0].z-colormap_g[0].y),
                              colormap_b[0].y + n2*(colormap_b[0].z-colormap_b[0].y), 1.f);
      }
      else {
              const float n3 = 3*norm-2;
              val = (float4) (colormap_r[0].z + n3*(colormap_r[0].w-colormap_r[0].z),
                              colormap_g[0].z + n3*(colormap_g[0].w-colormap_g[0].z),
                              colormap_b[0].z + n3*(colormap_b[0].w-colormap_b[0].z), 1.f);
      }
      write_imagef(texture_normuv, pixelcoord, val);
    }

// Bilinear interpolation of a 2D vector / a scalar in a cell ; vec_diff = position inside the cell, in [0, 1[ x [0, 1[.
float2 float2_bilinearinterpol(float2 vec_diff, float2 u00, float2 u01, float2 u10, float2 u11) {                
                
        float2 u0i = u00+vec_diff.y*(u01-u00); 
       
        return u0i+vec_diff.x*(u10+vec_diff.y*(u11-u10)-u0i);     
}

float scalar_bilinearinterpol(float2 vec_diff, float u00, float u01, float u10, float u11) {                
                
        float u0i = u00+vec_diff.y*(u01-u00); 
                
        return u0i+vec_diff.x*(u10+vec_diff.y*(u11-u10)-u0i);     
}

 
// Cubic Lagrange interpolation on a 4 x 4 stencil (um1m1 = u(-1,-1) ... u22 = u(2,2)) at position s inside the central cell.
float scalar_polyinterpol(float2 s,     float um1m1, float um10, float um11 , float um12,
                                        float u0m1 , float u00 , float u01  , float u02,
                                        float u1m1 , float u10 , float u11  , float u12,
                                        float u2m1 , float u20 , float u21  , float u22)
 {                        
    
        const float2 s2  = s*s ;
        const float2 s3  = s2*s ;
        const float2 w_m1 = -0.33333333333333333333f*s +0.5f*s2 -0.16666666666666666666f*s3;
        const float2 w_0  = 1-s2 + 0.5f*(s3-s);
        const float2 w_1  = s + 0.5f*(s2-s3);
        const float2 w_2  = 0.16666666666666666666f*(s3-s);     
      
        return w_m1.y*( w_m1.x*um1m1 + w_0.x*u0m1+w_1.x*u1m1 + w_2.x*u2m1) + w_0.y*( w_m1.x*um10 + w_0.x*u00+w_1.x*u10 + w_2.x*u20)
                + w_1.y*( w_m1.x*um11 + w_0.x*u01+w_1.x*u11 + w_2.x*u21)
                + w_2.y*( w_m1.x*um12 + w_0.x*u02+w_1.x*u12 + w_2.x*u22);       
}

float2 grid_percent_vec(float2 vec, int2 x1int) {
    return (float2) (vec.x - (float)x1int.x, vec.y - (float) x1int.y);
}
      
    
// Dye advection, semi-Lagrangian : follow the velocity backwards (3-stage Runge-Kutta) and interpolate the dye
// there with a bicubic (4 x 4 Lagrange) interpolation. Two copies of the dye (.x, .y) are read / written alternately.
__kernel void 
    dye_bipolyinter(
        __global float2 *dye,   
        __global float4 *uvrho,
        __global uchar *obstacle,
        uchar read_channel
         )
    { 
      const int2 pixelcoord = (int2) (get_global_id(0), get_global_id(1));  
      const float2 fpixelcoord = (float2) ((float) pixelcoord.x, (float) pixelcoord.y);
      
      const int ix = get_global_id(0);
      const int iy = get_global_id(1);             
      const int size_0 = get_global_size(0);
      const int size_1 = get_global_size(1);
      
      const int two_d_index = ix*size_1 + iy;
      
      if(obstacle[two_d_index]>0) return;
      
      float qn = 0.;      
             
      float2 k1 = (float2) (-uvrho[two_d_index].x,-uvrho[two_d_index].y);
      float2 xtemp = fpixelcoord + 0.5f*k1;      

      if(xtemp.x>=1 && xtemp.x<size_0-2 && xtemp.y>=1 && xtemp.y<size_1-2 ) {    
   
          int2 xint = (int2) ((int) xtemp.x, (int) xtemp.y);
          int indexm10 = size_1*(xint.x-1) + xint.y;
          int index00  = indexm10+size_1;
          int index10  = index00+size_1;
          int index20  = index10+size_1;     
        
          float2 k2;
          k2.x = - scalar_polyinterpol(grid_percent_vec(xtemp, xint), uvrho[indexm10-1].x, uvrho[indexm10].x, uvrho[indexm10+1].x, uvrho[indexm10+2].x,
                                                                      uvrho[index00-1].x,  uvrho[index00].x, uvrho[index00+1].x, uvrho[index00+2].x,
                                                                      uvrho[index10-1].x,  uvrho[index10].x, uvrho[index10+1].x, uvrho[index10+2].x,
                                                                      uvrho[index20-1].x,  uvrho[index20].x, uvrho[index20+1].x, uvrho[index20+2].x);
          k2.y = - scalar_polyinterpol(grid_percent_vec(xtemp, xint), uvrho[indexm10-1].y, uvrho[indexm10].y, uvrho[indexm10+1].y, uvrho[indexm10+2].y,
                                                                      uvrho[index00-1].y,  uvrho[index00].y, uvrho[index00+1].y, uvrho[index00+2].y,
                                                                      uvrho[index10-1].y,  uvrho[index10].y, uvrho[index10+1].y, uvrho[index10+2].y,
                                                                      uvrho[index20-1].y,  uvrho[index20].y, uvrho[index20+1].y, uvrho[index20+2].y);
                                                                                                                        
          xtemp = fpixelcoord + 0.75f*k2;
           
          if(xtemp.x>=1 && xtemp.x<size_0-2 && xtemp.y>=1 && xtemp.y<size_1-2 ) {    
              xint = (int2) ((int) xtemp.x, (int) xtemp.y);
              indexm10 = size_1*(xint.x-1) + xint.y;
              index00  = indexm10+size_1;
              index10  = index00+size_1;
              index20  = index10+size_1;  
      
              float2 k3;
              k3.x = - scalar_polyinterpol(grid_percent_vec(xtemp, xint), uvrho[indexm10-1].x, uvrho[indexm10].x, uvrho[indexm10+1].x, uvrho[indexm10+2].x,
                                                                      uvrho[index00-1].x,  uvrho[index00].x, uvrho[index00+1].x, uvrho[index00+2].x,
                                                                      uvrho[index10-1].x,  uvrho[index10].x, uvrho[index10+1].x, uvrho[index10+2].x,
                                                                      uvrho[index20-1].x,  uvrho[index20].x, uvrho[index20+1].x, uvrho[index20+2].x);
              k3.y = - scalar_polyinterpol(grid_percent_vec(xtemp, xint), uvrho[indexm10-1].y, uvrho[indexm10].y, uvrho[indexm10+1].y, uvrho[indexm10+2].y,
                                                                      uvrho[index00-1].y,  uvrho[index00].y, uvrho[index00+1].y, uvrho[index00+2].y,
                                                                      uvrho[index10-1].y,  uvrho[index10].y, uvrho[index10+1].y, uvrho[index10+2].y,
                                                                      uvrho[index20-1].y,  uvrho[index20].y, uvrho[index20+1].y, uvrho[index20+2].y);
                                                                    
          
              xtemp = fpixelcoord + 0.22222222222222222222f*k1 +0.33333333333333333333f *k2 + 0.44444444444444444444f*k3;
        
              if(xtemp.x>=1 && xtemp.x<size_0-2 && xtemp.y>=1 && xtemp.y<size_1-2 ) {  
                 
                  xint = (int2) ((int) xtemp.x, (int) xtemp.y);
                  indexm10 = size_1*(xint.x-1) + xint.y;
                  index00  = indexm10+size_1;
                  index10  = index00+size_1;
                  index20  = index10+size_1;    
                  
                  // Finally interpole the dye at point xtemp                  
                  if(read_channel==1)
                      qn = scalar_polyinterpol(grid_percent_vec(xtemp, xint), 
                                                                      dye[indexm10-1].y, dye[indexm10].y, dye[indexm10+1].y, dye[indexm10+2].y,
                                                                      dye[index00-1].y , dye[index00].y , dye[index00+1].y , dye[index00+2].y,
                                                                      dye[index10-1].y , dye[index10].y , dye[index10+1].y , dye[index10+2].y,
                                                                      dye[index20-1].y , dye[index20].y , dye[index20+1].y , dye[index20+2].y);
           
                  else qn = scalar_polyinterpol(grid_percent_vec(xtemp, xint), dye[indexm10-1].x, dye[indexm10].x, dye[indexm10+1].x, dye[indexm10+2].x,
                                                                      dye[index00-1].x, dye[index00].x, dye[index00+1].x, dye[index00+2].x,
                                                                      dye[index10-1].x, dye[index10].x, dye[index10+1].x, dye[index10+2].x,
                                                                      dye[index20-1].x, dye[index20].x, dye[index20+1].x, dye[index20+2].x);
              
              }
          }
      } 
      
       // Update the dye and the texture             
       if(read_channel==0) dye[two_d_index].y = qn;
       else dye[two_d_index].x = qn;    
    }
// Writes the dye value of every active emitter into the dye field and decrements its remaining life.
__kernel void 
    update_dye_emitter(
        __global float4 *dye_emitter,   
        __global float2 *dye
         )
    { 
       
      float4 emitter = dye_emitter[get_global_id(0)];
      if(emitter.w>0) {
          emitter.w = emitter.w-1.;
          dye[(int)emitter.x] = (float2) (emitter.z, emitter.z);
          dye_emitter[get_global_id(0)] = emitter;
      }      
             
   
    }
         
// Dye -> alpha channel of the texture. Two distinct images (read_only / write_only) :
// __read_write images are an optional OpenCL 2.0 feature that many devices (Intel CPU runtime) lack.
__kernel void 
    dye_to_texture(
        __global float2 *dye,   
        __read_only image2d_t texture_in,
        __write_only image2d_t texture_out,
        uchar read_channel
         )
    { 
      const int ix = get_global_id(0);
      const int iy = get_global_id(1);
      const int2 pixelcoord = (int2) (ix,iy);
      const int two_d_index = ix*get_global_size(1) + iy;
      
      const float qn = (read_channel==0) ? dye[two_d_index].x : dye[two_d_index].y; 
      float4 res = read_imagef(texture_in, pixelcoord);
      res.w = 1-qn;
      write_imagef(texture_out, pixelcoord, res);    
    }

__kernel void 
    copy_texture(
        __read_only image2d_t texture_in,
        __write_only image2d_t texture_out
         )
    {
      const int2 pixelcoord = (int2) (get_global_id(0), get_global_id(1));
      write_imagef(texture_out, pixelcoord, read_imagef(texture_in, pixelcoord));
    }
