# -*- coding: utf-8 -*-
"""LBM D2Q9 / MRT in native Python (NumPy, no OpenCL).

Same model, conventions and boundary conditions as the OpenCL solver (lbm_opencl.py + kernels.cl),
so both can be compared field by field :
  - arrays are indexed [x, y] ; populations are (9, nx, ny)
  - D2Q9 order : rest, +x, +y, -x, -y, (1,1), (-1,1), (-1,-1), (1,-1)
  - uLB = 0.04 * speedup, nu = uLB * l_ref / Re, omega = 1 / (3 nu + 0.5)
  - MRT collision with the same relaxation rates and the same equilibrium moments as the `moments` kernel
  - boundaries : inlet (x = 0) with imposed velocity, outlet (x = nx-1) zero gradient,
    top / bottom (y = 0, ny-1) periodic for the unknown populations, bounce-back on obstacles
It is much slower than the OpenCL solver : meant as a readable reference.
"""
import numpy as np

# Lattice velocities, weights and opposite directions (bounce-back)
C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int8)
T = np.array([4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.])
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# Moments (rho, e, eps, jx, qx, jy, qy, pxx, pxy) of the populations
M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
              [-4, -1, -1, -1, -1, 2, 2, 2, 2],
              [4, -2, -2, -2, -2, 1, 1, 1, 1],
              [0, 1, 0, -1, 0, 1, -1, -1, 1],
              [0, -2, 0, 2, 0, 1, -1, -1, 1],
              [0, 0, 1, 0, -1, 1, 1, -1, -1],
              [0, 0, -2, 0, 2, 1, 1, -1, -1],
              [0, 1, -1, 1, -1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=np.float64)


class LBMNative:
    def __init__(self, re, l_ref, nx, ny, speedup=1., dtype=np.float32):
        """
        re (float)      : Reynolds number
        l_ref (float)   : reference length in cells (the OpenCL solver uses ny/2)
        nx, ny (int)    : grid size
        speedup (float) : multiplies the lattice velocity (uLB = 0.04 * speedup)
        """
        self.re, self.l_ref, self.nx, self.ny, self.speedup = re, l_ref, nx, ny, speedup
        self.dtype = dtype
        self.uLB = 0.04 * speedup
        self.obstacle = np.zeros((nx, ny), dtype=bool)
        self.update_re(re)

        # Macroscopic fields (kept from the last collision ; obstacle cells keep their initial value)
        self.u = np.full((nx, ny), self.uLB, dtype=dtype)
        self.v = np.zeros((nx, ny), dtype=dtype)
        self.rho = np.ones((nx, ny), dtype=dtype)
        self.fin = self._equilibrium(self.rho, self.u, self.v).astype(dtype)
        self.iteration = 0

    # ------------------------------------------------------------------ parameters
    def update_re(self, re):
        """Change the Reynolds number : recompute the viscosity and the MRT collision matrix."""
        self.re = re
        self.nulb = self.uLB * self.l_ref / re
        self.omega = 1.0 / (3. * self.nulb + 0.5)
        # Relaxation rates : conserved moments (rho, jx, jy) 0, e and eps 1.4, qx and qy 1.2, stress omega
        s = np.array([0., 1.4, 1.4, 0., 1.2, 0., 1.2, self.omega, self.omega])
        self.M = M.astype(self.dtype)
        self.M1S = (np.linalg.inv(M) @ np.diag(s)).astype(self.dtype)

    def set_obstacle(self, obstacle):
        """obstacle : boolean array (nx, ny), True on solid cells."""
        self.obstacle = np.asarray(obstacle, dtype=bool).reshape(self.nx, self.ny)

    # ------------------------------------------------------------------ one step
    def run(self, nb_iter):
        for _ in range(nb_iter):
            self._bcs()
            self._inlet()
            fout = self._collision()
            self._streaming(fout)
            self.iteration += 1

    def get_fields(self):
        """(u, v, rho) copies, arrays (nx, ny)."""
        return self.u.copy(), self.v.copy(), self.rho.copy()

    # ------------------------------------------------------------------ steps
    def _equilibrium(self, rho, ux, uy):
        cu = 3. * (C[:, 0, None, None] * ux + C[:, 1, None, None] * uy)
        usqr = 1.5 * (ux * ux + uy * uy)
        return T[:, None, None] * rho * (1. + cu + 0.5 * cu * cu - usqr)

    def _bcs(self):
        """Populations coming from outside the domain : periodic top/bottom, zero gradient on the outlet."""
        f = self.fin
        f[[4, 7, 8], :, -1] = f[[4, 7, 8], :, 0]       # top wall    <- bottom row
        f[[2, 5, 6], :, 0] = f[[2, 5, 6], :, -1]       # bottom wall <- top row
        f[[3, 6, 7], -1, :] = f[[3, 6, 7], -2, :]      # outlet      <- previous column

    def _inlet(self):
        """Left wall : velocity (uLB, 0) imposed, density from the known populations,
        unknown populations (1, 5, 8) set to equilibrium."""
        f = self.fin[:, 0, :]
        ux = self.uLB
        rho = (2. * (f[3] + f[7] + f[6]) + f[0] + f[4] + f[2]) / (1. - ux)
        for i in (1, 5, 8):
            cu = 3. * C[i, 0] * ux
            f[i] = T[i] * rho * (1. + cu + 0.5 * cu * cu - 1.5 * ux * ux)

    def _collision(self):
        """MRT collision on fluid cells, bounce-back on obstacles."""
        f = self.fin.reshape(9, -1)
        m = self.M @ f
        rho, jx, jy = m[0], m[3], m[5]

        # Equilibrium moments, as in the `moments` kernel (j / rho approximated by j)
        j2 = 3. * (jx * jx + jy * jy)
        delta = np.zeros_like(m)
        delta[1] = m[1] + 2. * rho - j2
        delta[2] = m[2] - rho + j2
        delta[4] = m[4] + jx
        delta[6] = m[6] + jy
        delta[7] = m[7] - jx * jx + jy * jy
        delta[8] = m[8] - jx * jy
        fout = f - self.M1S @ delta

        solid = self.obstacle.ravel()
        fout[:, solid] = f[OPPOSITE][:, solid]

        fluid = ~solid
        self.u.ravel()[fluid] = (jx / rho)[fluid]
        self.v.ravel()[fluid] = (jy / rho)[fluid]
        self.rho.ravel()[fluid] = rho[fluid]
        return fout.reshape(self.fin.shape)

    def _streaming(self, fout):
        """Move each population along its velocity ; nothing enters from outside (filled by _bcs / _inlet)."""
        fin = self.fin
        fin[:] = 0.
        nx, ny = self.nx, self.ny
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            dst = (slice(max(cx, 0), nx + min(cx, 0)), slice(max(cy, 0), ny + min(cy, 0)))
            src = (slice(max(-cx, 0), nx + min(-cx, 0)), slice(max(-cy, 0), ny + min(-cy, 0)))
            fin[i][dst] = fout[i][src]
