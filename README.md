# LBM-PyOpenCL

Lattice Boltzmann Method with Multiple Relaxation Time (LBM-MRT), D2Q9, in 2D.

The same model is implemented three times, and driven from a web page:
  - **OpenCL** (`lbm_opencl.py` + `kernels.cl`) : on a GPU, or on the CPU with an OpenCL runtime ;
  - **Python / NumPy** (`lbm_native.py`) : readable reference implementation ;
  - **Python / Numba** (`lbm_numba.py`) : the same, compiled and multithreaded.
Tests check that the three give the same fields (`tests/`).

# How to run it ?
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py        # opens http://localhost:8000
```
The page lets you choose the solver (and the GPU / CPU device), the grid, the Reynolds number, the obstacle and
the display. It shows the FPS and where the solver runs (GPU, or CPU on N threads).

# How to change the obstacle shape ?
Obstacle shapes are text files (one `x;y` point per line, closed contour) in `world_objects/`.
Drop a new file there and pick it in the page.

# Documentation
See [DOCUMENTATION.md](DOCUMENTATION.md) : overview, reading guide of the code, LBM primer, file by file,
client/server protocol, tests, performance.

# Tests
```
.venv\Scripts\python -m unittest discover -s tests -v
```

# Some videos
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Ph7dVfcWveQ/0.jpg)](https://www.youtube.com/watch?v=Ph7dVfcWveQ)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/w-3g8Jeuhn4/0.jpg)](https://www.youtube.com/watch?v=w-3g8Jeuhn4)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Brx27BIAaQw/0.jpg)](https://www.youtube.com/watch?v=Brx27BIAaQw)

