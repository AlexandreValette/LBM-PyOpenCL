# -*- coding: utf-8 -*-
"""LBM D2Q9 / MRT in Python, compiled and multithreaded with Numba.

Same model and conventions as lbm_native.py (NumPy) and lbm_opencl.py : only the way of computing differs.
Collision, streaming and dye advection are explicit loops over the cells, parallelized over x with prange.
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
def _lagrange(s):
    """Weights of the 4 samples (offsets -1, 0, 1, 2) for a position s in [0, 1[ inside the central cell."""
    s2 = s * s
    s3 = s2 * s
    return (-s / 3. + 0.5 * s2 - s3 / 6.,
            1. - s2 + 0.5 * (s3 - s),
            s + 0.5 * (s2 - s3),
            (s3 - s) / 6.)


@njit(cache=True)
def _interp(f, px, py):
    """Bicubic (4 x 4 Lagrange) interpolation of the 2D array f at (px, py), which must be inside the grid."""
    ix = int(px)
    iy = int(py)
    wx = _lagrange(px - ix)
    wy = _lagrange(py - iy)
    total = 0.
    for a in range(4):
        for b in range(4):
            total += wx[a] * wy[b] * f[ix + a - 1, iy + b - 1]
    return total


@njit(parallel=True, cache=True)
def _dye_advect(src, dst, u, v, obstacle, nx, ny):
    """Semi-Lagrangian advection (kernel dye_bipolyinter) : 3-stage Runge-Kutta backwards along the velocity,
    then bicubic interpolation of the dye. A trajectory point outside the interior of the grid gives 0."""
    for x in prange(nx):
        for y in range(ny):
            if obstacle[x, y]:
                continue
            qn = 0.
            k1x = -u[x, y]
            k1y = -v[x, y]
            px = x + 0.5 * k1x
            py = y + 0.5 * k1y
            if px >= 1 and px < nx - 2 and py >= 1 and py < ny - 2:
                k2x = -_interp(u, px, py)
                k2y = -_interp(v, px, py)
                px = x + 0.75 * k2x
                py = y + 0.75 * k2y
                if px >= 1 and px < nx - 2 and py >= 1 and py < ny - 2:
                    k3x = -_interp(u, px, py)
                    k3y = -_interp(v, px, py)
                    px = x + (2. / 9.) * k1x + (1. / 3.) * k2x + (4. / 9.) * k3x
                    py = y + (2. / 9.) * k1y + (1. / 3.) * k2y + (4. / 9.) * k3y
                    if px >= 1 and px < nx - 2 and py >= 1 and py < ny - 2:
                        qn = _interp(src, px, py)
            dst[x, y] = qn


@njit(cache=True)
def _emit(dye, emitters):
    """Write the dye of the active emitters (rows : ix, iy, value, life), then decrement their life."""
    for e in range(emitters.shape[0]):
        if emitters[e, 3] > 0:
            ix = int(emitters[e, 0])
            iy = int(emitters[e, 1])
            dye[0, ix, iy] = emitters[e, 2]
            dye[1, ix, iy] = emitters[e, 2]
            emitters[e, 3] -= 1


@njit(cache=True)
def _run(nb_iter, fin, fout, u, v, rhof, obstacle, m1s, nx, ny, u_lb, use_dye, dye, dye_read, emitters):
    """nb_iter time steps. Returns the index of the dye copy that holds the latest dye field."""
    for _ in range(nb_iter):
        _bcs(fin, nx, ny)
        _inlet(fin, ny, u_lb)
        _collision(fin, fout, u, v, rhof, obstacle, m1s, nx, ny)
        _streaming(fout, fin, nx, ny)
        if use_dye:
            _dye_advect(dye[dye_read], dye[1 - dye_read], u, v, obstacle, nx, ny)
            dye_read = 1 - dye_read
            _emit(dye, emitters)
    return dye_read


class LBMNumba(LBMNative):
    """Same interface as LBMNative : run(n), get_fields(), set_obstacle(mask), update_re(re), and the dye
    (enable_dye, add_dye_rect, get_dye, inherited)."""
    def __init__(self, re, l_ref, nx, ny, speedup=1., threads=0):
        """threads : number of Numba threads (0 = all the cores). The setting is global to the process."""
        super().__init__(re, l_ref, nx, ny, speedup=speedup, dtype=np.float32)
        self.threads = int(threads) if threads else numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(max(1, min(self.threads, numba.config.NUMBA_NUM_THREADS)))
        self.fout = np.zeros_like(self.fin)

    def run(self, nb_iter):
        use_dye = self.dye is not None
        dye = self.dye if use_dye else np.zeros((2, 1, 1), dtype=np.float32)
        self._dye_read = _run(int(nb_iter), self.fin, self.fout, self.u, self.v, self.rho, self.obstacle,
                              np.ascontiguousarray(self.M1S), self.nx, self.ny, float(self.uLB),
                              use_dye, dye, self._dye_read, self._emitters)
        self.iteration += int(nb_iter)


def warm_up():
    """Compile (or load from the cache) the kernels, so the first real step is not slow."""
    for dye in (False, True):
        model = LBMNumba(1e3, 4, 8, 8)
        if dye:
            model.enable_dye()
            model.add_dye_rect(1, 2, 2, 4, 1., 5)
        model.run(1)
