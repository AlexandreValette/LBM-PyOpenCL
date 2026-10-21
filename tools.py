# -*- coding: utf-8 -*-
"""Obstacle tools : load a profile file (x;y per line), scale / rotate / move it, rasterize it on the grid."""
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
        
    def load_file(self, scaling=0.2, file="world_objects/NACA 4412.txt"):
        """Load a profile file (one `x;y` point per line, a closed contour), center it and scale it so that its
        largest dimension is `scaling` * nx cells. The result is kept in self.data (float points)."""
        
        data = np.loadtxt(file, delimiter = ";")  # Load object
        data[:,0] = data[:,0]-np.mean(data[:,0])  # Center it
        data[:,1] = data[:,1]-np.mean(data[:,1]) 
        
        dx = np.max(data[:,0])-np.min(data[:,0])  # Move bottom left to position 0,0
        dy = np.max(data[:,1])-np.min(data[:,1])
        data = data / max(dy,dx)        
        data = data*self.nx*scaling  # Scale it
        
        self.data = data        
    
    def rotate_and_move(self, angle=0, x=0, y=0)  :
        """Rotate the loaded profile by `angle` degrees, move it to the cell (x, y) and rasterize it.
        returns (obstacle, data_int) : the boolean mask (nx, ny), True inside the profile, and the
        integer coordinates of the contour."""
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)        
        R = np.matrix([[c, -s], [s, c]]) 
        data = np.dot(self.data, R)
        data[:,0] +=x
        data[:,1] +=y
        
        data_int = data.astype(np.int32)        
        bbPath = mplPath.Path(data_int)
        # Clip the bounding box to the grid, then test all its cells at once
        min_x = max(np.min(data_int[:,0])-1, 0)
        max_x = min(np.max(data_int[:,0])+1, self.nx)
        min_y = max(np.min(data_int[:,1])-1, 0)
        max_y = min(np.max(data_int[:,1])+1, self.ny)
        self.obstacle[:,:]=False
        if max_x > min_x and max_y > min_y:
            xs, ys = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y), indexing='ij')
            points = np.column_stack([xs.ravel(), ys.ravel()])
            self.obstacle[min_x:max_x, min_y:max_y] = bbPath.contains_points(points).reshape(xs.shape)
                
        return self.obstacle , data_int
            
        



# ---------------------------------------------------------------------- rendering (NumPy)
# Same 4-point color map as the OpenCL display kernels (blue -> white -> red)
COLORMAP = np.array([[0.2298057,   0.298717966, 0.753683153],
                     [0.66725292,  0.77917646,  0.99295],
                     [0.968203399, 0.720844,    0.61229299],
                     [0.705673158, 0.01555616,  0.1502328]])
OBSTACLE_COLOR = (0., 0.5, 0.)


def render_field(u, v, obstacle, mode="norm", vmin=0., vmax=0.1, dye=None):
    """Color image of a velocity field, like the OpenCL display kernels.

    u, v, obstacle : arrays (nx, ny)
    mode           : "norm" (velocity norm, from vmin to vmax) or "curl" (signed curl, -vmax to +vmax)
    dye            : optional dye field (nx, ny) : it fades the colors to the dark background (alpha = 1 - dye),
                     like dye_to_texture in the OpenCL solver
    returns        : uint8 array (ny, nx, 3), first row = top (y axis pointing up)
    """
    if mode == "curl":
        curl = np.zeros_like(u)
        curl[:-1, :-1] = (v[1:, :-1] - v[:-1, :-1]) - (u[:-1, 1:] - u[:-1, :-1])
        x = 0.5 + 0.5 * curl / vmax
    else:
        x = (np.sqrt(u * u + v * v) - vmin) / (vmax - vmin)
    x = np.clip(x, 0., 1.)
    knots = np.linspace(0., 1., len(COLORMAP))
    rgb = np.stack([np.interp(x, knots, COLORMAP[:, k]) for k in range(3)], axis=-1)
    rgb[obstacle] = OBSTACLE_COLOR
    if dye is not None:
        alpha = (1. - dye)[..., None]
        rgb = np.clip(rgb * alpha + 0.01 * (1. - alpha), 0., 1.)
    return (rgb.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)
