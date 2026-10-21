# -*- coding: utf-8 -*-
"""Runs one LBM simulation in a background thread and produces JPEG frames.

Two interchangeable solvers (config["solver"]) :
  - "opencl" : lbm_opencl.py + kernels.cl, on a GPU or an OpenCL CPU runtime
  - "native" : lbm_native.py, the same model in plain NumPy
  - "numba"  : lbm_numba.py, the same model compiled and multithreaded with Numba
No window / OpenGL here : each backend gives an RGB image, which is JPEG encoded. A client
(web page, later Unity...) only has to ask for frames and display them.
"""
import io
import os
import threading
import time
import traceback
from collections import deque

import numpy as np
from PIL import Image

import tools
import ocl_driver

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OBSTACLE_DIR = os.path.join(BASE_DIR, "world_objects")

# Parameters that can be changed while the simulation is running
LIVE_KEYS = ("re", "iter_per_frame", "display", "t_min", "t_max", "curl_max", "dye_y", "dye_freq", "dye_color", "collision", "inlet", "outlet", "smagorinsky",
             "obstacle_file", "obstacle_x", "obstacle_y", "obstacle_rotation", "obstacle_scale")


def _default_device():
    """Index (in ocl_driver.list_devices()) of the first GPU, else of the first device."""
    devices = ocl_driver.list_devices()
    return next((d["id"] for d in devices if d["type"] == "GPU"), 0)


DEFAULT_CONFIG = {
    "solver": "opencl",       # "opencl", "native" (NumPy : slow, use a small grid) or "numba"
    "threads": 0,             # Numba solver : number of threads (0 = all cores)
    "device": _default_device(),   # OpenCL device : index in ocl_driver.list_devices()
    "nx": 512, "ny": 256,     # rounded to a multiple of 32
    "speedup": 1.0,
    "re": 1e4,
    "iter_per_frame": 10,
    "display": "norm",        # "norm" (velocity norm) or "curl"
    "t_min": 0.0, "t_max": 0.08, "curl_max": 0.03,
    "obstacle_file": "FlatPlane.txt",   # file of world_objects/, "" for none
    "obstacle_x": 0.25, "obstacle_y": 0.5,   # position, as a fraction of the grid
    "obstacle_rotation": 45.0, "obstacle_scale": 0.14,
    "dye": True, "dye_y": 0.5, "dye_freq": 10, "dye_color": "#030303",   # passive dye : emitter at the inlet, refreshed every dye_freq frames
    # Numerical methods (see lbm_native.py) : the first of each list is the original one
    "collision": "smagorinsky",     # "mrt", "regularized" or "smagorinsky" (default : stable up to Re = 1e7, see DOCUMENTATION.md)
    "inlet": "equilibrium",         # "equilibrium" or "zouhe"
    "outlet": "zero_gradient",      # "zero_gradient" or "convective"
    "smagorinsky": 0.1,             # constant Cs of the Smagorinsky collision
    "jpeg_quality": 85,
}


def list_obstacles():
    """Profile files (x;y per line) available in world_objects/."""
    if not os.path.isdir(OBSTACLE_DIR):
        return []
    return sorted(f for f in os.listdir(OBSTACLE_DIR) if f.lower().endswith(".txt"))


def _dye_rgb(c):
    """Dye color "#rrggbb" -> (r, g, b) in 0..1."""
    h = str(c["dye_color"]).lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255. for i in (0, 2, 4))


def _dye_rect(c):
    """Dye emitter at the inlet : (cx_min, cx_max, cy_min, cy_max, value, life in steps), 3 cells high around dye_y."""
    cy = int(float(c["dye_y"]) * c["ny"])
    return (1, 2, max(cy - 1, 1), min(cy + 2, c["ny"] - 1), 1., 100)


class ObstacleBuilder:
    """Boolean obstacle mask (nx, ny) from the obstacle_* parameters (profile files are cached)."""
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        self._cache = {}

    def build(self, c):
        obstacle = np.zeros((self.nx, self.ny), dtype=bool)
        name = c["obstacle_file"]
        if name:
            path = os.path.join(OBSTACLE_DIR, os.path.basename(name))
            scaling = float(c["obstacle_scale"])
            key = (path, scaling)
            if key not in self._cache:
                om = tools.ObsertacleManager(self.nx, self.ny)
                om.load_file(scaling=scaling, file=path)
                self._cache[key] = om
            obstacle, _ = self._cache[key].rotate_and_move(
                angle=float(c["obstacle_rotation"]),
                x=int(float(c["obstacle_x"]) * self.nx), y=int(float(c["obstacle_y"]) * self.ny))
        return np.ascontiguousarray(obstacle)


class OpenCLBackend:
    """lbm_opencl.py on an OpenCL device. Display kernels write into an OpenCL image (no OpenGL sharing)."""
    def __init__(self, c):
        import pyopencl as cl
        import lbm_opencl
        self._cl = cl
        self._lbm_opencl = lbm_opencl
        self.c = c
        nx, ny = c["nx"], c["ny"]
        is_cpu = ocl_driver.list_devices()[int(c["device"])]["type"] == "CPU"
        exp = 4 if is_cpu else 5
        self.ocl_d = ocl_driver.OCL_Driver(device_index=int(c["device"]), interop=False,
                                           workers_exp1=exp, workers_exp2=exp)
        self.device_text = self.ocl_d.describe()
        self.queue = self.ocl_d.queue

        self.lbm = lbm_opencl.LBM(float(c["re"]), ny / 2, nx, ny, speedup=float(c["speedup"]))
        self.lbm_ocl = lbm_opencl.LBM_OCL(self.ocl_d)
        self.lbm_ocl.init_data(self.lbm)
        self._obstacle = np.zeros((nx, ny), dtype=bool)
        self._apply_methods()

        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
        self.image = cl.create_image(self.ocl_d.ctx, cl.mem_flags.READ_WRITE, fmt, shape=(nx, ny))
        self._host_image = np.zeros((ny, nx, 4), dtype=np.float32)
        self.lbm_ocl.add_postrun_texture_interop(self.image, value="norm")
        self._set_display(c["display"])

        self.dye_rects, self._counter, self._emit_dye = [], 0, 0
        if c["dye"]:
            self.lbm_ocl.enable_dye(np.zeros((nx, ny, 2), dtype=np.float32))
            self.dye_emitters = lbm_opencl.DyeEmitter(self.lbm_ocl.shape2D)
            self.dye_rects.append(lbm_opencl.RectDye(*_dye_rect(c)))

    def set_obstacle(self, obstacle):
        self._obstacle = obstacle
        self.lbm_ocl.update_obstacle(obstacle)

    def _apply_methods(self):
        c = self.c
        self.lbm_ocl.set_methods(c["collision"], c["inlet"], c["outlet"], float(c["smagorinsky"]))

    def diagnostics_fields(self):
        """(u, v, rho, solid) read back from the device, for the stability indicators."""
        uvrho = self.lbm_ocl.get_rho_u_v()
        return uvrho[:, :, 0], uvrho[:, :, 1], uvrho[:, :, 2], self._obstacle

    def tau(self):
        return 1. / self.lbm.omega

    def update(self, old):
        c = self.c
        if any(c[k] != old[k] for k in ("collision", "inlet", "outlet", "smagorinsky")):
            self._apply_methods()
        if c["re"] != old["re"]:
            self.lbm_ocl.update_re(float(c["re"]))
        if c["dye_y"] != old["dye_y"] and self.dye_rects:
            # Forget the emitters of the old position (they would keep emitting up to 100 steps), emit now
            self.dye_rects[0] = self._lbm_opencl.RectDye(*_dye_rect(c))
            self.dye_emitters = self._lbm_opencl.DyeEmitter(self.lbm_ocl.shape2D)
            self._emit_dye = self._counter
        if c["display"] != old["display"]:
            self._set_display(c["display"])
        if any(c[k] != old[k] for k in ("t_min", "t_max", "curl_max", "display")):
            self._set_minmax()

    def step(self, iters):
        if self.dye_rects and self._counter == self._emit_dye:
            for rect in self.dye_rects:
                self.dye_emitters.add_rect_emitter(rect)
            self.lbm_ocl.add_dye_emitters(self.dye_emitters)
            self._emit_dye += max(1, int(self.c["dye_freq"]))
        self.lbm_ocl.run(iters)
        self.queue.finish()
        self._counter += 1

    def render(self):
        self._cl.enqueue_copy(self.queue, self._host_image, self.image,
                              origin=(0, 0), region=(self.c["nx"], self.c["ny"])).wait()
        rgb, alpha = self._host_image[:, :, :3], self._host_image[:, :, 3:4]
        # The dye lives in the alpha channel : it fades the color towards the dye color
        img = np.clip(rgb * alpha + np.asarray(_dye_rgb(self.c)) * (1. - alpha), 0., 1.)
        return (img[::-1] * 255).astype(np.uint8)       # y axis points up

    def _set_display(self, mode):
        # post_run_functions[0] is the field -> image conversion (dye, if any, comes after)
        fn = self.lbm_ocl._curl_to_texture if mode == "curl" else self.lbm_ocl._norm_uv_to_texture
        self.lbm_ocl.post_run_functions[0] = fn
        self._set_minmax()

    def _set_minmax(self):
        c = self.c
        if c["display"] == "curl":
            self.lbm_ocl.set_texture_min_max(0., float(c["curl_max"]))
        else:
            self.lbm_ocl.set_texture_min_max(float(c["t_min"]), float(c["t_max"]))


class NativeBackend:
    """lbm_native.py : the same model in plain NumPy, on the CPU. Subclasses change the model (_make_model)."""
    def __init__(self, c):
        self.c = c
        self.lbm = self._make_model()
        self.device_text = self._device_text()
        self._apply_methods()
        self._dye_rect, self._counter, self._emit_dye = None, 0, 0
        if c["dye"]:
            self.lbm.enable_dye()
            self._dye_rect = _dye_rect(c)

    def _make_model(self):
        import lbm_native
        c = self.c
        return lbm_native.LBMNative(float(c["re"]), c["ny"] / 2, c["nx"], c["ny"], speedup=float(c["speedup"]))

    def _device_text(self):
        return "CPU : Python natif (NumPy)"

    def set_obstacle(self, obstacle):
        self.lbm.set_obstacle(obstacle)

    def _apply_methods(self):
        c = self.c
        self.lbm.set_methods(c["collision"], c["inlet"], c["outlet"], float(c["smagorinsky"]))

    def diagnostics_fields(self):
        """(u, v, rho, solid) for the stability indicators."""
        u, v, rho = self.lbm.get_fields()
        return u, v, rho, self.lbm.obstacle

    def tau(self):
        return 1. / self.lbm.omega

    def update(self, old):
        if any(self.c[k] != old[k] for k in ("collision", "inlet", "outlet", "smagorinsky")):
            self._apply_methods()
        if self.c["re"] != old["re"]:
            self.lbm.update_re(float(self.c["re"]))
        if self._dye_rect and self.c["dye_y"] != old["dye_y"]:
            self._dye_rect = _dye_rect(self.c)
            self.lbm._emitters = np.zeros((0, 4), dtype=np.float32)    # drop the emitters of the old position
            self._emit_dye = self._counter

    def step(self, iters):
        if self._dye_rect and self._counter == self._emit_dye:      # same schedule as OpenCLBackend
            self.lbm.add_dye_rect(*self._dye_rect)
            self._emit_dye += max(1, int(self.c["dye_freq"]))
        self.lbm.run(iters)
        self._counter += 1

    def render(self):
        c = self.c
        u, v, _ = self.lbm.get_fields()
        dye = self.lbm.get_dye() if self.lbm.dye is not None else None
        k = float(c["speedup"])          # same scaling as LBM_OCL.set_texture_min_max
        if c["display"] == "curl":
            return tools.render_field(u, v, self.lbm.obstacle, "curl", vmax=float(c["curl_max"]) * k, dye=dye,
                                     dye_color=_dye_rgb(c))
        return tools.render_field(u, v, self.lbm.obstacle, "norm",
                                  vmin=float(c["t_min"]) * k, vmax=float(c["t_max"]) * k, dye=dye,
                                  dye_color=_dye_rgb(c))


class NumbaBackend(NativeBackend):
    """lbm_numba.py : same model, compiled and multithreaded with Numba."""
    def _make_model(self):
        import lbm_numba
        c = self.c
        lbm_numba.warm_up()
        return lbm_numba.LBMNumba(float(c["re"]), c["ny"] / 2, c["nx"], c["ny"],
                                  speedup=float(c["speedup"]), threads=int(c["threads"]))

    def _device_text(self):
        return "CPU : Python Numba sur %d threads" % self.lbm.threads


BACKENDS = {"opencl": OpenCLBackend, "native": NativeBackend, "numba": NumbaBackend}


class SimulationSession:
    def __init__(self, config=None):
        self.config = dict(DEFAULT_CONFIG)
        self.config.update(config or {})
        c = self.config
        c["nx"] = max(32, int(round(float(c["nx"]) / 32)) * 32)
        c["ny"] = max(32, int(round(float(c["ny"]) / 32)) * 32)

        self.state = "starting"       # starting / running / stopped / error
        self.error = None
        self.device_text = ""
        self.iteration = 0
        self.diagnostics = {}         # tau, mach (max), rho_mean : refreshed twice a second

        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending = {}
        self._frame_requested = threading.Event()
        self._frame = (0, b"")
        self._loop_times = deque(maxlen=200)

    # ---------------------------------------------------------------- public API
    def start(self):
        self._thread = threading.Thread(target=self._run, name="lbm-session", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def set_params(self, params):
        """Change live parameters (see LIVE_KEYS) ; applied before the next batch of steps."""
        with self._lock:
            self._pending.update({k: v for k, v in params.items() if k in LIVE_KEYS})

    def request_frame(self):
        self._frame_requested.set()

    def get_frame(self):
        """(frame_id, jpeg_bytes) of the last rendered frame."""
        return self._frame

    def stats(self):
        now = time.perf_counter()
        times = [t for t in self._loop_times if now - t < 1.0]
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
        iters = self.config["iter_per_frame"]
        nx, ny = self.config["nx"], self.config["ny"]
        return {"type": "stats", "state": self.state, "error": self.error,
                "fps": round(fps, 1), "mlups": round(fps * iters * nx * ny / 1e6, 1),
                "iteration": self.iteration, "device": self.device_text,
                "grid": [nx, ny], **self.diagnostics}

    # ---------------------------------------------------------------- worker thread
    def _run(self):
        try:
            c = self.config
            backend = BACKENDS[c["solver"]](c)
            self.device_text = backend.device_text
            obstacles = ObstacleBuilder(c["nx"], c["ny"])
            backend.set_obstacle(obstacles.build(c))
            self.state = "running"
            last_diagnostics = 0.
            while not self._stop.is_set():
                self._apply_pending(backend, obstacles)
                iters = int(c["iter_per_frame"])
                backend.step(iters)
                self.iteration += iters
                now = time.perf_counter()
                self._loop_times.append(now)
                if now - last_diagnostics > 0.5:
                    last_diagnostics = now
                    self._check_stability(backend)
                if self._frame_requested.is_set():
                    self._frame_requested.clear()
                    self._frame = (self._frame[0] + 1, self._jpeg(backend.render()))
            self.state = "stopped"
        except Exception as exc:
            traceback.print_exc()
            self.error = "%s: %s" % (type(exc).__name__, exc)
            self.state = "error"

    def _check_stability(self, backend):
        """Stability indicators : relaxation time tau, maximum Mach number (u_max / cs, cs = 1/sqrt(3)) and mean density.
        Stops the simulation with an explicit error when the fields are no longer finite (the run has diverged)."""
        u, v, rho, solid = backend.diagnostics_fields()
        fluid = ~solid
        speed, density = np.hypot(u, v)[fluid], rho[fluid]
        if not (np.isfinite(speed).all() and np.isfinite(density).all()) or density.min() <= 0.:
            raise ArithmeticError("la simulation a divergé (valeurs non finies) : baisser le Reynolds ou choisir une "
                                  "collision plus stable (régularisée / Smagorinsky)")
        self.diagnostics = {"tau": round(backend.tau(), 6), "mach": round(float(speed.max()) * 3 ** 0.5, 3),
                            "rho_mean": round(float(density.mean()), 5)}

    def _apply_pending(self, backend, obstacles):
        with self._lock:
            pending, self._pending = self._pending, {}
        if not pending:
            return
        old = dict(self.config)
        self.config.update(pending)
        try:
            backend.update(old)
            if any(self.config[k] != old[k] for k in LIVE_KEYS if k.startswith("obstacle")):
                backend.set_obstacle(obstacles.build(self.config))
        except Exception as exc:
            traceback.print_exc()
            self.error = "%s: %s" % (type(exc).__name__, exc)

    def _jpeg(self, rgb):
        buf = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(buf, format="JPEG", quality=int(self.config["jpeg_quality"]))
        return buf.getvalue()


if __name__ == "__main__":
    # Headless smoke test : python simulation_session.py [opencl|native|numba]
    import sys
    config = {"solver": sys.argv[1] if len(sys.argv) > 1 else "opencl"}
    if config["solver"] in ("native", "numba"):
        config.update(nx=256, ny=128, iter_per_frame=2)
    session = SimulationSession(config)
    session.start()
    t0 = time.time()
    while time.time() - t0 < 8 and session.state in ("starting", "running"):
        session.request_frame()
        time.sleep(0.25)
    print(session.stats())
    frame_id, jpeg = session.get_frame()
    print("frame", frame_id, len(jpeg), "bytes")
    open("smoke_test.jpg", "wb").write(jpeg)
    session.stop()
