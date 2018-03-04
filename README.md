# LBM-PyOpenCL

Lattice Boltzmann Method with Multiple Relaxation Time (LBM-MRT), D2Q9.
Run on OpenCL compatible GPU.
Tested on NVIDIA GTX970.

Code : 
  - Host : PYTHON 
  - Computation : OPENCL on GPU (or CPU)
  - Graphics : OPENGL (with OpenCL interop)
 
# How to run it ?
You can use the notebook Run.ipynb or the script Run.py.

# How to change the obstacle shape ?
Obstacle shape are read from a text file. See Circle.txt for example.

# Third parties libraries
You need :
  - numpy ;
  - matplotlib ;
  - pyopencl (and opencl) ;
  - pyopengl ;
  - sys ;
  - time ;
  
For installing pyopencl, see :
https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_PyOpenCl_On_Anaconda_For_Windows?lang=en

# Some videos
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Ph7dVfcWveQ/0.jpg)](https://www.youtube.com/watch?v=Ph7dVfcWveQ)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/w-3g8Jeuhn4/0.jpg)](https://www.youtube.com/watch?v=w-3g8Jeuhn4)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Brx27BIAaQw/0.jpg)](https://www.youtube.com/watch?v=Brx27BIAaQw)

