# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.path as mplPath

class ObsertacleManager:
    def __init__(self, nx, ny):
        """
        nx (integer) : x size of grid
        ny (integer) : y size of grid
        """        
        self.nx = nx
        self.ny = ny
        # Matrix for obstacles
        self.obstacle = np.zeros((nx, ny), dtype=bool)
        
    def load_file(self, scaling=0.2, file="NACA 4412.geo"):
        
        data = np.loadtxt(file, delimiter = ";")  # Load object
        data[:,0] = data[:,0]-np.mean(data[:,0])  # Center it
        data[:,1] = data[:,1]-np.mean(data[:,1]) 
        
        dx = np.max(data[:,0])-np.min(data[:,0])  # Move bottom left to position 0,0
        dy = np.max(data[:,1])-np.min(data[:,1])
        data = data / max(dy,dx)        
        data = data*self.nx*scaling  # Scale it
        
        self.data = data        
    
    def rotate_and_move(self, angle=0, x=0, y=0)  :
        """
        return the obstacle matrix and obstacles coordinates
        """
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)        
        R = np.matrix([[c, -s], [s, c]]) 
        data = np.dot(self.data, R)
        data[:,0] +=x
        data[:,1] +=y
        
        data_int = data.astype(np.int32)        
        bbPath = mplPath.Path(data_int)
        min_x = np.min(data_int[:,0])-1
        max_x = np.max(data_int[:,0])+1
        min_y = np.min(data_int[:,1])-1
        max_y = np.max(data_int[:,1])+1
        self.obstacle[:,:]=False
        for i in range(min_x, max_x):
            for j in range(min_y, max_y):
                self.obstacle[i,j] = bbPath.contains_point((i, j))
                
        return self.obstacle , data_int
            
        

