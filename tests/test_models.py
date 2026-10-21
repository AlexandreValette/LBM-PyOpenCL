# -*- coding: utf-8 -*-
"""Consistency tests between the three implementations of the same LBM model.

    python -m unittest discover -s tests -v

  - lbm_native (NumPy) is the readable reference ;
  - lbm_numba must match it ;
  - lbm_opencl (GPU / OpenCL CPU) must match it too (skipped when no OpenCL device works).
All of them use float32, so fields agree to about 1e-6, not to the last bit. The dye is more sensitive (it
follows sharp fronts, and over hundreds of steps tiny float differences grow) : it is compared over a single
advection step on a random field (strict), and over a few dozen steps of a real simulation (looser).
"""
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lbm_native
import lbm_numba
import lbm_opencl
import ocl_driver
import simulation_session as ss

NX, NY, RE = 128, 64, 1e3
TOLERANCE = 5e-6          # max absolute difference on u, v, rho (all about 0.04 and 1)
STEPS = (1, 10, 100, 300)
DYE_STEPS = (1, 10, 30)
DYE_TOLERANCE = 1e-4      # dye values are between 0 and about 1.3
DYE_EMITTER = (1, 2, 31, 34, 1., 1000)     # cells x in [1, 2[, y in [31, 34[ , value 1, life 1000 steps


def setUpModule():
    # unittest re-enables all warnings : silence the noisy pyopencl ones
    import pyopencl
    warnings.filterwarnings("ignore", category=pyopencl.RepeatedKernelRetrieval)
    warnings.filterwarnings("ignore", category=pyopencl.CompilerWarning)


def make_obstacle():
    """The default flat plate, rotated 45 degrees, on the test grid."""
    config = dict(ss.DEFAULT_CONFIG, nx=NX, ny=NY)
    return ss.ObstacleBuilder(NX, NY).build(config)


def max_differences(fields_a, fields_b, fluid):
    """Largest |difference| of (u, v, rho) over the fluid cells (obstacle cells are never updated)."""
    return [float(np.abs(a - b)[fluid].max()) for a, b in zip(fields_a, fields_b)]


class NativeModel(unittest.TestCase):
    def test_uniform_flow_stays_uniform(self):
        """Without obstacle, the initial uniform flow (uLB, 0) is a steady solution."""
        model = lbm_native.LBMNative(RE, NY / 2, NX, NY)
        model.run(50)
        u, v, rho = model.get_fields()
        self.assertLess(np.abs(u - model.uLB).max(), 1e-5)
        self.assertLess(np.abs(v).max(), 1e-5)
        self.assertLess(np.abs(rho - 1.).max(), 1e-5)

    def test_obstacle_slows_the_flow(self):
        """An obstacle creates a wake : velocities are no longer uniform."""
        model = lbm_native.LBMNative(RE, NY / 2, NX, NY)
        model.set_obstacle(make_obstacle())
        model.run(100)
        u, v, _ = model.get_fields()
        self.assertGreater(np.abs(v).max(), 1e-3)

    def test_obstacle_mask_is_not_empty(self):
        mask = make_obstacle()
        self.assertEqual(mask.shape, (NX, NY))
        self.assertGreater(mask.sum(), 10)


class NumbaMatchesNumpy(unittest.TestCase):
    def test_fields_match_at_several_times(self):
        obstacle = make_obstacle()
        reference = lbm_native.LBMNative(RE, NY / 2, NX, NY)
        compiled = lbm_numba.LBMNumba(RE, NY / 2, NX, NY)
        reference.set_obstacle(obstacle)
        compiled.set_obstacle(obstacle)
        done = 0
        for target in STEPS:
            reference.run(target - done)
            compiled.run(target - done)
            done = target
            diffs = max_differences(reference.get_fields(), compiled.get_fields(), ~obstacle)
            self.assertLess(max(diffs), TOLERANCE, "step %d : %s" % (target, diffs))

    def test_changing_reynolds_matches(self):
        obstacle = make_obstacle()
        reference = lbm_native.LBMNative(RE, NY / 2, NX, NY)
        compiled = lbm_numba.LBMNumba(RE, NY / 2, NX, NY)
        for model in (reference, compiled):
            model.set_obstacle(obstacle)
            model.run(50)
            model.update_re(5e3)
            model.run(50)
        diffs = max_differences(reference.get_fields(), compiled.get_fields(), ~obstacle)
        self.assertLess(max(diffs), TOLERANCE, diffs)


def smooth_flow_and_random_dye():
    """A velocity field with some structure, and a random dye field (sharp, a hard case for the interpolation)."""
    xx, yy = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    u = (0.04 + 0.03 * np.sin(xx / 9.) * np.cos(yy / 7.)).astype(np.float32)
    v = (0.03 * np.cos(xx / 11.) * np.sin(yy / 5.)).astype(np.float32)
    dye = np.random.default_rng(1).random((NX, NY)).astype(np.float32)
    return u, v, dye


class DyeNumbaMatchesNumpy(unittest.TestCase):
    def test_one_advection_step_on_random_dye(self):
        obstacle = make_obstacle()
        u, v, dye = smooth_flow_and_random_dye()
        reference = lbm_native.LBMNative(RE, NY / 2, NX, NY)
        reference.set_obstacle(obstacle)
        reference.u[:], reference.v[:] = u, v
        reference.enable_dye()
        reference.dye[0] = dye
        reference._advect_dye()
        src, dst = np.stack([dye, np.zeros_like(dye)]), np.zeros((NX, NY), dtype=np.float32)
        lbm_numba._dye_advect(src[0], dst, u, v, obstacle, NX, NY)
        diff = np.abs(reference.get_dye() - dst)[~obstacle].max()
        self.assertLess(diff, 1e-5)

    def test_simulation_with_dye(self):
        obstacle = make_obstacle()
        reference = lbm_native.LBMNative(RE, NY / 2, NX, NY)
        compiled = lbm_numba.LBMNumba(RE, NY / 2, NX, NY)
        for model in (reference, compiled):
            model.set_obstacle(obstacle)
            model.enable_dye()
            model.add_dye_rect(*DYE_EMITTER)
        done = 0
        for target in DYE_STEPS:
            reference.run(target - done)
            compiled.run(target - done)
            done = target
            diff = np.abs(reference.get_dye() - compiled.get_dye())[~obstacle].max()
            self.assertLess(diff, DYE_TOLERANCE, "step %d" % target)
        self.assertGreater(reference.get_dye().max(), 0.5)       # the dye really spreads from the emitter

    def test_dye_stays_zero_without_emitter(self):
        model = lbm_native.LBMNative(RE, NY / 2, NX, NY)
        model.enable_dye()
        model.run(20)
        self.assertEqual(float(np.abs(model.get_dye()).max()), 0.)


def opencl_available():
    try:
        return len(ocl_driver.list_devices()) > 0
    except Exception:
        return False


@unittest.skipUnless(opencl_available(), "no working OpenCL device")
class OpenCLMatchesNumpy(unittest.TestCase):
    def test_every_opencl_device_matches(self):
        obstacle = make_obstacle()
        for device in ocl_driver.list_devices():
            with self.subTest(device="%s : %s" % (device["type"], device["name"])):
                config = dict(ss.DEFAULT_CONFIG, nx=NX, ny=NY, re=RE, dye=False, collision="mrt", device=device["id"])
                backend = ss.OpenCLBackend(config)
                backend.set_obstacle(obstacle)
                reference = lbm_native.LBMNative(RE, NY / 2, NX, NY)
                reference.set_obstacle(obstacle)
                done = 0
                for target in STEPS:
                    reference.run(target - done)
                    backend.lbm_ocl.run(target - done)
                    backend.queue.finish()
                    done = target
                    uvrho = backend.lbm_ocl.get_rho_u_v()          # (nx, ny, 4) : u, v, rho, unused
                    opencl_fields = (uvrho[:, :, 0], uvrho[:, :, 1], uvrho[:, :, 2])
                    diffs = max_differences(reference.get_fields(), opencl_fields, ~obstacle)
                    self.assertLess(max(diffs), TOLERANCE, "step %d : %s" % (target, diffs))


@unittest.skipUnless(opencl_available(), "no working OpenCL device")
class DyeOpenCLMatchesNumpy(unittest.TestCase):
    def test_every_opencl_device_matches(self):
        obstacle = make_obstacle()
        u, v, dye = smooth_flow_and_random_dye()
        reference = lbm_native.LBMNative(RE, NY / 2, NX, NY)
        reference.set_obstacle(obstacle)
        reference.u[:], reference.v[:] = u, v
        reference.enable_dye()
        reference.dye[0] = dye
        reference._advect_dye()
        for device in ocl_driver.list_devices():
            with self.subTest(device="%s : %s" % (device["type"], device["name"])):
                import pyopencl as cl
                config = dict(ss.DEFAULT_CONFIG, nx=NX, ny=NY, re=RE, dye=False, collision="mrt", device=device["id"])
                backend = ss.OpenCLBackend(config)
                solver = backend.lbm_ocl
                solver.update_obstacle(obstacle)
                uvrho = np.zeros((NX, NY, 4), dtype=np.float32)
                uvrho[:, :, 0], uvrho[:, :, 1], uvrho[:, :, 2] = u, v, 1.
                cl.enqueue_copy(solver.ocl_d.queue, solver.ocl_uvrho, uvrho)
                solver.enable_dye(np.stack([dye, np.zeros_like(dye)], axis=-1).copy())
                solver.dye()                                     # one advection step
                backend.queue.finish()
                diff = np.abs(reference.get_dye() - solver.get_dye())[~obstacle].max()
                self.assertLess(diff, 1e-5)

    def test_simulation_with_dye_matches_for_a_few_steps(self):
        obstacle = make_obstacle()
        for device in ocl_driver.list_devices():
            with self.subTest(device="%s : %s" % (device["type"], device["name"])):
                config = dict(ss.DEFAULT_CONFIG, nx=NX, ny=NY, re=RE, dye=False, collision="mrt", device=device["id"])
                backend = ss.OpenCLBackend(config)
                backend.set_obstacle(obstacle)
                solver = backend.lbm_ocl
                solver.enable_dye(np.zeros((NX, NY, 2), dtype=np.float32))
                emitters = lbm_opencl.DyeEmitter(solver.shape2D)
                emitters.add_rect_emitter(lbm_opencl.RectDye(*DYE_EMITTER))
                solver.add_dye_emitters(emitters)
                reference = lbm_native.LBMNative(RE, NY / 2, NX, NY)
                reference.set_obstacle(obstacle)
                reference.enable_dye()
                reference.add_dye_rect(*DYE_EMITTER)
                done = 0
                for target in DYE_STEPS:
                    reference.run(target - done)
                    solver.run(target - done)
                    backend.queue.finish()
                    done = target
                    diff = np.abs(reference.get_dye() - solver.get_dye())[~obstacle].max()
                    self.assertLess(diff, DYE_TOLERANCE, "step %d" % target)


METHODS = [(c, i, o) for c in lbm_native.COLLISIONS for i in lbm_native.INLETS for o in lbm_native.OUTLETS]
METHOD_STEPS = (1, 50, 300)


def reference_after(methods, steps, obstacle, re=RE):
    model = lbm_native.LBMNative(re, NY / 2, NX, NY)
    model.set_methods(*methods)
    model.set_obstacle(obstacle)
    model.run(steps)
    return model.get_fields()


class MethodsNumbaMatchesNumpy(unittest.TestCase):
    def test_every_method_combination(self):
        """Regularized / Smagorinsky collisions, Zou-He inlet, convective outlet : Numba = NumPy."""
        obstacle = make_obstacle()
        for methods in METHODS:
            with self.subTest(methods=methods):
                model = lbm_numba.LBMNumba(RE, NY / 2, NX, NY)
                model.set_methods(*methods)
                model.set_obstacle(obstacle)
                done = 0
                for target in METHOD_STEPS:
                    model.run(target - done)
                    done = target
                    diffs = max_differences(reference_after(methods, target, obstacle), model.get_fields(), ~obstacle)
                    self.assertLess(max(diffs), TOLERANCE * 4, "step %d : %s" % (target, diffs))

    def test_unknown_method_is_refused(self):
        model = lbm_native.LBMNative(RE, NY / 2, NX, NY)
        with self.assertRaises(ValueError):
            model.set_methods(collision="bgk")

    def test_methods_stay_stable_at_high_reynolds(self):
        """Re = 1e6 on a small grid : regularized and Smagorinsky collisions stay finite and bounded."""
        obstacle = make_obstacle()
        for methods in (("regularized", "zouhe", "convective"), ("smagorinsky", "zouhe", "convective")):
            with self.subTest(methods=methods):
                u, v, rho = reference_after(methods, 600, obstacle, re=1e6)
                self.assertTrue(np.isfinite(u).all() and np.isfinite(rho).all())
                self.assertLess(np.abs(u).max(), 0.3)


@unittest.skipUnless(opencl_available(), "no working OpenCL device")
class MethodsOpenCLMatchesNumpy(unittest.TestCase):
    def test_every_method_combination_on_every_device(self):
        obstacle = make_obstacle()
        for device in ocl_driver.list_devices():
            config = dict(ss.DEFAULT_CONFIG, nx=NX, ny=NY, re=RE, dye=False, collision="mrt", device=device["id"])
            backend = ss.OpenCLBackend(config)
            backend.set_obstacle(obstacle)
            for methods in METHODS:
                with self.subTest(device="%s : %s" % (device["type"], device["name"]), methods=methods):
                    solver = lbm_opencl.LBM_OCL(backend.ocl_d)
                    solver.init_data(lbm_opencl.LBM(RE, NY / 2, NX, NY))
                    solver.update_obstacle(obstacle)
                    solver.set_methods(*methods)
                    done = 0
                    for target in METHOD_STEPS:
                        solver.run(target - done)
                        backend.queue.finish()
                        done = target
                        uvrho = solver.get_rho_u_v()
                        fields = (uvrho[:, :, 0], uvrho[:, :, 1], uvrho[:, :, 2])
                        diffs = max_differences(reference_after(methods, target, obstacle), fields, ~obstacle)
                        self.assertLess(max(diffs), TOLERANCE * 4, "step %d : %s" % (target, diffs))


if __name__ == "__main__":
    unittest.main()
