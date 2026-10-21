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
  - optional passive dye, advected like the OpenCL one (kernels dye_bipolyinter / update_dye_emitter)
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

        # Passive dye (see enable_dye) : two copies of the field, read / written alternately
        self.dye = None
        self._dye_read = 0
        self._emitters = np.zeros((0, 4), dtype=np.float32)   # rows : (ix, iy, value, remaining life)

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

    # ------------------------------------------------------------------ dye
    def enable_dye(self):
        """Activate the passive dye (starts at 0 everywhere). Does not act on the fluid."""
        self.dye = np.zeros((2, self.nx, self.ny), dtype=np.float32)
        self._dye_read = 0

    def add_dye_rect(self, cx_min, cx_max, cy_min, cy_max, value, life):
        """Add a rectangle of emitting cells [cx_min, cx_max[ x [cy_min, cy_max[. Each one writes `value` in the
        dye field at every time step during `life` steps (same as lbm_opencl.RectDye)."""
        xs, ys = np.meshgrid(np.arange(cx_min, cx_max), np.arange(cy_min, cy_max), indexing="ij")
        new = np.zeros((xs.size, 4), dtype=np.float32)
        new[:, 0], new[:, 1], new[:, 2], new[:, 3] = xs.ravel(), ys.ravel(), value, life
        alive = self._emitters[self._emitters[:, 3] > 0]
        self._emitters = np.vstack([alive, new])

    def get_dye(self):
        """The dye field (nx, ny), float32 copy."""
        return self.dye[self._dye_read].copy()

    def _emit_dye(self):
        """Write the dye of the active emitters, then decrement their life (kernel update_dye_emitter)."""
        e = self._emitters
        alive = e[:, 3] > 0
        if alive.any():
            self.dye[:, e[alive, 0].astype(int), e[alive, 1].astype(int)] = e[alive, 2]
            e[alive, 3] -= 1

    def _advect_dye(self):
        """Semi-Lagrangian advection (kernel dye_bipolyinter) : for each fluid cell, follow the velocity backwards
        with a 3-stage Runge-Kutta, then read the dye there with a bicubic (4 x 4 Lagrange) interpolation.
        If a trajectory point leaves the interior of the grid, the new dye is 0."""
        nx, ny = self.nx, self.ny
        src, dst = self.dye[self._dye_read], self.dye[1 - self._dye_read]
        x = np.arange(nx, dtype=np.float32)[:, None] * np.ones((1, ny), dtype=np.float32)
        y = np.arange(ny, dtype=np.float32)[None, :] * np.ones((nx, 1), dtype=np.float32)
        uv = np.stack([self.u.ravel(), self.v.ravel()])          # (2, nx*ny), interpolated together
        k1x, k1y = -self.u, -self.v

        px, py = x + 0.5 * k1x, y + 0.5 * k1y
        ok = self._inside(px, py)
        k2x, k2y = -self._interpolate(uv, px, py)

        px, py = x + 0.75 * k2x, y + 0.75 * k2y
        ok &= self._inside(px, py)
        k3x, k3y = -self._interpolate(uv, px, py)

        px = x + (2. / 9.) * k1x + (1. / 3.) * k2x + (4. / 9.) * k3x
        py = y + (2. / 9.) * k1y + (1. / 3.) * k2y + (4. / 9.) * k3y
        ok &= self._inside(px, py)
        qn = self._interpolate(src.reshape(1, -1), px, py)[0]

        fluid = ~self.obstacle                     # obstacle cells are not updated
        dst[fluid] = np.where(ok, qn, 0.)[fluid]
        self._dye_read = 1 - self._dye_read

    def _inside(self, px, py):
        """True where the 4 x 4 interpolation stencil fits in the grid (same test as the kernel)."""
        return (px >= 1) & (px < self.nx - 2) & (py >= 1) & (py < self.ny - 2)

    def _interpolate(self, fields, px, py):
        """Cubic Lagrange interpolation at the points (px, py) (arrays (nx, ny)) of fields : array (k, nx*ny),
        returns an array (k, nx, ny). Points outside the valid range are clamped (the result is then meaningless :
        the caller masks it)."""
        ix = np.clip(px.astype(np.intp), 1, self.nx - 3)
        iy = np.clip(py.astype(np.intp), 1, self.ny - 3)
        wx = self._lagrange_weights(px - ix.astype(px.dtype))
        wy = self._lagrange_weights(py - iy.astype(py.dtype))
        base = (ix * self.ny + iy).ravel()                      # flat index of the cell holding each point
        out = np.zeros((fields.shape[0], base.size), dtype=np.float32)
        for a in range(4):
            for b in range(4):
                weight = (wx[a] * wy[b]).ravel()
                out += weight * fields.take(base + ((a - 1) * self.ny + (b - 1)), axis=1)
        return out.reshape((fields.shape[0], self.nx, self.ny))

    @staticmethod
    def _lagrange_weights(s):
        """Weights of the 4 samples (offsets -1, 0, 1, 2) for a position s in [0, 1[ inside the central cell."""
        s2 = s * s
        s3 = s2 * s
        return (-s / 3. + 0.5 * s2 - s3 / 6.,
                1. - s2 + 0.5 * (s3 - s),
                s + 0.5 * (s2 - s3),
                (s3 - s) / 6.)

    # ------------------------------------------------------------------ one step
    def run(self, nb_iter):
        for _ in range(nb_iter):
            self._bcs()
            self._inlet()
            fout = self._collision()
            self._streaming(fout)
            if self.dye is not None:
                self._advect_dye()
                self._emit_dye()
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
