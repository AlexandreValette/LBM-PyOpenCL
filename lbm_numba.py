# -*- coding: utf-8 -*-
"""LBM D2Q9 / MRT in Python, compiled and multithreaded with Numba.

Same model and conventions as lbm_native.py (NumPy) and lbm_opencl.py : only the way of computing differs.
Collision and streaming are explicit loops over the cells, parallelized over x with prange.
"""
import numpy as np
import numba
from numba import njit, prange

from lbm_native import LBMNative, C, T, OPPOSITE

_CX = C[:, 0].astype(np.int64)
_CY = C[:, 1].astype(np.int64)
_OPP = OPPOSITE.astype(np.int64)


@njit(cache=True)
def _bcs(fin, nx, ny):
    """Populations coming from outside the domain : periodic top/bottom, zero gradient on the outlet."""
    for x in range(nx):
        fin[4, x, ny - 1] = fin[4, x, 0]
        fin[7, x, ny - 1] = fin[7, x, 0]
        fin[8, x, ny - 1] = fin[8, x, 0]
        fin[2, x, 0] = fin[2, x, ny - 1]
        fin[5, x, 0] = fin[5, x, ny - 1]
        fin[6, x, 0] = fin[6, x, ny - 1]
    for y in range(ny):
        fin[3, nx - 1, y] = fin[3, nx - 2, y]
        fin[6, nx - 1, y] = fin[6, nx - 2, y]
        fin[7, nx - 1, y] = fin[7, nx - 2, y]


@njit(cache=True)
def _inlet(fin, ny, u_lb):
    """Left wall : velocity (u_lb, 0) imposed, density from the known populations, populations 1, 5, 8 at equilibrium."""
    usqr = 1.5 * u_lb * u_lb
    for y in range(ny):
        rho = (2. * (fin[3, 0, y] + fin[7, 0, y] + fin[6, 0, y])
               + fin[0, 0, y] + fin[4, 0, y] + fin[2, 0, y]) / (1. - u_lb)
        for i in (1, 5, 8):
            cu = 3. * _CX[i] * u_lb
            fin[i, 0, y] = T[i] * rho * (1. + cu + 0.5 * cu * cu - usqr)


@njit(parallel=True, cache=True)
def _collision(fin, fout, u, v, rhof, obstacle, m1s, nx, ny):
    """MRT collision on fluid cells, bounce-back on obstacles (same formulas as the `moments` kernel)."""
    for x in prange(nx):
        f = np.empty(9)
        delta = np.empty(9)
        for y in range(ny):
            for i in range(9):
                f[i] = fin[i, x, y]
            if obstacle[x, y]:
                for i in range(9):
                    fout[i, x, y] = f[_OPP[i]]
                continue
            rho = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]
            m1 = -4. * f[0] - (f[1] + f[2] + f[3] + f[4]) + 2. * (f[5] + f[6] + f[7] + f[8])
            m2 = 4. * f[0] - 2. * (f[1] + f[2] + f[3] + f[4]) + (f[5] + f[6] + f[7] + f[8])
            jx = f[1] - f[3] + f[5] - f[6] - f[7] + f[8]
            m4 = -2. * f[1] + 2. * f[3] + f[5] - f[6] - f[7] + f[8]
            jy = f[2] - f[4] + f[5] + f[6] - f[7] - f[8]
            m6 = -2. * f[2] + 2. * f[4] + f[5] + f[6] - f[7] - f[8]
            m7 = f[1] - f[2] + f[3] - f[4]
            m8 = f[5] - f[6] + f[7] - f[8]
            u[x, y] = jx / rho
            v[x, y] = jy / rho
            rhof[x, y] = rho

            j2 = 3. * (jx * jx + jy * jy)
            delta[0] = 0.
            delta[1] = m1 + 2. * rho - j2
            delta[2] = m2 - rho + j2
            delta[3] = 0.
            delta[4] = m4 + jx
            delta[5] = 0.
            delta[6] = m6 + jy
            delta[7] = m7 - jx * jx + jy * jy
            delta[8] = m8 - jx * jy
            for i in range(9):
                s = 0.
                for j in range(9):
                    s += m1s[i, j] * delta[j]
                fout[i, x, y] = f[i] - s


@njit(parallel=True, cache=True)
def _streaming(fout, fin, nx, ny):
    """Pull each population from the neighbour it comes from ; nothing enters from outside the domain."""
    for x in prange(nx):
        for y in range(ny):
            for i in range(9):
                sx = x - _CX[i]
                sy = y - _CY[i]
                if sx >= 0 and sx < nx and sy >= 0 and sy < ny:
                    fin[i, x, y] = fout[i, sx, sy]
                else:
                    fin[i, x, y] = 0.


@njit(cache=True)
def _run(nb_iter, fin, fout, u, v, rhof, obstacle, m1s, nx, ny, u_lb):
    for _ in range(nb_iter):
        _bcs(fin, nx, ny)
        _inlet(fin, ny, u_lb)
        _collision(fin, fout, u, v, rhof, obstacle, m1s, nx, ny)
        _streaming(fout, fin, nx, ny)


class LBMNumba(LBMNative):
    """Same interface as LBMNative : run(n), get_fields(), set_obstacle(mask), update_re(re)."""
    def __init__(self, re, l_ref, nx, ny, speedup=1., threads=0):
        """threads : number of Numba threads (0 = all the cores). The setting is global to the process."""
        super().__init__(re, l_ref, nx, ny, speedup=speedup, dtype=np.float32)
        self.threads = int(threads) if threads else numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(max(1, min(self.threads, numba.config.NUMBA_NUM_THREADS)))
        self.fout = np.zeros_like(self.fin)

    def run(self, nb_iter):
        _run(int(nb_iter), self.fin, self.fout, self.u, self.v, self.rho, self.obstacle,
             np.ascontiguousarray(self.M1S), self.nx, self.ny, float(self.uLB))
        self.iteration += int(nb_iter)


def warm_up():
    """Compile (or load from the cache) the kernels, so the first real step is not slow."""
    LBMNumba(1e3, 4, 8, 8).run(1)
