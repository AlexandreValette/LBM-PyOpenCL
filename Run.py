# -*- coding: utf-8 -*-
import LBM_pygame

# The id of the GPU plateform. If unkown, just try 0, 1, 2 until it works.
# PS : It will work on CPU (but very slow)
# PS : The name of the selected plateform will be displayed in the console
platform_id = 2 

nx = 512
ny  = 256
speedup=1
re = 1e4
scale = 2

game = LBM_pygame.LBM_pygame(platform_id, nx = nx, ny = ny, scale=scale, re = re, speedup=speedup)
game.add_obstacle(filename="FlatPlane.txt", x=nx/4, y = ny/2, rotation=45, scaling=0.14)
game.display(True)

game.enable_dye(dye_freq=10/speedup)
game.add_dye_emitter(cx_min = 1, cx_max=2,  cy_min=int(ny/2), cy_max=int(ny/2)+3, value=1., life_dt=100 )

game.run(10) # Run 10 time steps per frame