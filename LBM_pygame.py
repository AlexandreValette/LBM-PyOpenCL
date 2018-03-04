# -*- coding: utf-8 -*-
import pygame
import LBM
import numpy as np
import pyopencl as cl
import Obstacle_manager as Obstacle_manager
import ocl_driver
# OpenGL imports
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData

class LBM_pygame:
    def __init__(self, platform_id, nx = 1024, ny = 512, scale =1, re = 1e7, speedup = 0.25):

        self.nx, self.ny, self.re, self.speedup = nx, ny, re, speedup
        self.win_width = int(nx*scale)
        self.win_height = int(ny*scale)
        self.is_dye = False
        self.temporal_re = False
        # Ini windows and openGl via pygame *********************************************************************************************
        pygame.init()
        screen = pygame.display.set_mode((self.win_width, self.win_height),  pygame.OPENGL|pygame.DOUBLEBUF)
        self.obstacle = np.zeros((nx,ny), dtype = np.bool)
        
        # Load a LBM
        ocl_d = ocl_driver.OCL_Driver(platform_id, interop=True)
        self.lbm = LBM.LBM(re, ny/2, nx, ny,  speedup = speedup)
        self.lbm_ocl = LBM.LBM_OCL(ocl_d)
        self._init_OGL()
        self.lbm_ocl.init_data(self.lbm)

    def _init_OGL(self):
        # Create an openGL texture
        glEnable(GL_TEXTURE_2D);
        glGenerateMipmap(GL_TEXTURE_2D);
        textureID = glGenTextures(1);
        glBindTexture( GL_TEXTURE_2D, textureID );# Bind it
        # Create a texture array and send it to opengl
        m_texture = np.zeros((self.nx,self.ny,4), dtype = np.float32) # An nx*ny*4 array
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);    # 1D texture
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);    # 2D texture
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA ,self.nx,self.ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_texture);

        # Setting the projection
        glMatrixMode(GL_PROJECTION);
        glOrtho(0, self.win_width, 0, self.win_height, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        # Set the opengl blend option
        glClearColor(0.01,0.01,0.01,1.)    ; 
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND);
        # Create OpenCL memory object from gl texture
        self.textureMem =cl.GLTexture( self.lbm_ocl.ocl_d.ctx, cl.mem_flags.READ_WRITE, GL_TEXTURE_2D, 0, textureID, 2);
        
    def add_obstacle(self, filename = "NACA 4412.geo", rotation=0, x=0, y=0, scaling=0.2):
        om = Obstacle_manager.ObsertacleManager(self.nx, self.ny)
        om.load_file( scaling=scaling, file = filename)
        obs, profil = om.rotate_and_move(angle = rotation, x=int(x), y = int(y))
        self.obstacle += obs 
    
    def add_random_obstacles(self, proba_random_obstacle = 0.0):
        if(proba_random_obstacle!=0.):
            obstacle = np.random.choice([False, True], size=(self.nx,self.ny), p =[1.-proba_random_obstacle,proba_random_obstacle])
            n=2
            obstacle[0:n,:]=False
            obstacle[self.nx-n:self.nx,:]=False
            obstacle[:,0:n]=False
            obstacle[:,self.ny-n:self.ny]=False            
            self.obstacle += obstacle

    def enable_dye(self, dye_freq=100, dye= None):
        self.dye_freq=dye_freq
        self.dye_emitters= LBM.DyeEmitter(self.lbm_ocl.shape2D)
        if(dye is None): dye = np.zeros((self.nx*self.ny*2), dtype = np.float32).reshape(self.nx,self.ny,2)   
        self.lbm_ocl.enable_dye(dye)
        self.is_dye = True
        self.rect_dye=[]
        
    def add_dye_emitter(self, cx_min = 1, cx_max=3,  cy_min=-1, cy_max=-1, value=1, life_dt=1): 
        if(cy_max==-1): cy_max=self.ny-2
        if(cy_min==-1): cy_min=2
        self.rect_dye.append(LBM.RectDye( int(cx_min),int(cx_max), int(cy_min), int(cy_max), value, life_dt))     
       
    
    def display(self, display_norm = True, t_min=0.0,t_max=0.08):
        
        if(display_norm):
            self.lbm_ocl.add_postrun_texture_interop(self.textureMem, value="norm")
            self.lbm_ocl.set_texture_min_max(t_min=t_min,t_max=t_max)
        else:
            self.lbm_ocl.add_postrun_texture_interop(self.textureMem, value="curl")
            self.lbm_ocl.set_texture_min_max(t_min=t_min,t_max=t_max)
    
    def _draw_plane(self):    
        glClear(GL_COLOR_BUFFER_BIT); 
        glBegin(GL_QUADS);
        glTexCoord2i(0, 0); glVertex2i(0, 0);
        glTexCoord2i(0, 1); glVertex2i(0, self.win_height);
        glTexCoord2i(1, 1); glVertex2i(self.win_width, self.win_height);
        glTexCoord2i(1, 0); glVertex2i(self.win_width, 0);
        glEnd();
        
    def increase_re(self,start_at=1,  re_from = 1, re_to = 1e5, fac = 1.5, freq = 10):
        self.start_at=start_at
        self.temporal_re = True
        self.re_from = re_from; self.re_to = re_to; self.fac = fac; self.freq = freq
        self.re = re_from
        self.lbm_ocl.update_re(self.re)
        
    def run(self, iter_per_frame = 10)   :     
        
        self.lbm_ocl.update_obstacle(self.obstacle)
        clock = pygame.time.Clock()    
        counter=-1 ; running = True ; zoom=1.;  emit_dye=0
        try:
            while running:
                clock.tick()  ;  
                counter+=1  
                
                # Manage events
                for event in pygame.event.get():            
                    if event.type == pygame.QUIT:
                        running = False  
                    if event.type == pygame.MOUSEBUTTONDOWN:            
                        if (event.button==5 or event.button==4):
                            old_zoom = zoom
                            if (event.button==5) : zoom *=1.3
                            else:          zoom *=0.7          
                            glMatrixMode(GL_PROJECTION);
                            glLoadIdentity();
                            glOrtho(0, self.win_width*zoom, 0, self.win_height*zoom, -1, 1);       
                            glMatrixMode(GL_MODELVIEW);                             
                            (mouseX, mouseY) = pygame.mouse.get_pos()
                            glTranslatef(self.win_width*zoom/2-mouseX*old_zoom,mouseY*zoom-self.win_height*old_zoom/2,0.)
                        if (event.button==1):
                            (mouseX, mouseY) = pygame.mouse.get_pos()
                            glTranslatef(self.win_width*zoom/2-mouseX*zoom,mouseY*zoom-self.win_height*zoom/2,0.)
                        if (event.button==3): 
                            ''                 
                
                if(self.temporal_re ):
                     if(counter%(self.freq/iter_per_frame)==0 and self.re<self.re_to and counter>(self.start_at/iter_per_frame)):
                         self.re = self.re*self.fac
                         self.lbm_ocl.update_re(self.re)                       
        
                # Manage dye emitters                
                if(self.is_dye and counter==emit_dye): 
                        
                        for rec in self.rect_dye:                            
                            self.dye_emitters.add_rect_emitter(rec)
                        self.lbm_ocl.add_dye_emitters(self.dye_emitters)
                        emit_dye+=np.int(self.dye_freq)                     
     
                # Run LBM and draw plane
                self.lbm_ocl.run(iter_per_frame);
                self._draw_plane()  ;        
                pygame.display.flip();
               
            pygame.quit()
            print("result: "+str(clock.get_fps())+" FPS")
        except SystemExit:
            pygame.quit()
            print("result: "+str(clock.get_fps())+" FPS")



