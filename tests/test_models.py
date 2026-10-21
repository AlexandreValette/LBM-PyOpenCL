# -*- coding: utf-8 -*-
"""Consistency tests between the three implementations of the same LBM model.

    python -m unittest discover -s tests -v

  - lbm_native (NumPy) is the readable reference ;
  - lbm_numba must match it ;
  - lbm_opencl (GPU / OpenCL CPU) must match it too (skipped when no OpenCL device works).
All of them use float32, so fields agree to about 1e-6, not to the last bit.
"""
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lbm_native
import lbm_numba
import ocl_driver
import simulation_session as ss

NX, NY, RE = 128, 64, 1e3
TOLERANCE = 5e-6          # max absolute difference on u, v, rho (all about 0.04 and 1)
STEPS = (1, 10, 100, 300)


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
                config = dict(ss.DEFAULT_CONFIG, nx=NX, ny=NY, re=RE, dye=False, device=device["id"])
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


if __name__ == "__main__":
    unittest.main()
