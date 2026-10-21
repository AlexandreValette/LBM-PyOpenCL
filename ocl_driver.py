# -*- coding: utf-8 -*-
"""OpenCL device selection, context and kernel compilation.

  - list_devices() : the usable OpenCL devices (GPU, CPU runtime), each tested with a trivial kernel ;
  - OCL_Driver     : context + command queue + compiled kernels (kernels.cl) of one device.
It also declares the optional Intel CPU runtime (intel-opencl-rt) to the OpenCL loader.
"""
import os
import sys

# Optional CPU runtime : Intel OpenCL runtime (pip install intel-opencl-rt, works on AMD CPUs too).
# The OpenCL loader bundled with the pyopencl wheel only knows the vendors registered in the Windows
# registry (admin rights needed), so the runtime library is declared through OCL_ICD_FILENAMES
# (the documented env var of the Khronos loader), before pyopencl is imported.
_INTEL_BIN = os.path.join(sys.prefix, "Library", "bin")
_INTEL_DLL = os.path.join(_INTEL_BIN, "intelocl64.dll")
if sys.platform == "win32" and os.path.isfile(_INTEL_DLL):
    os.add_dll_directory(_INTEL_BIN)
    os.environ["PATH"] = _INTEL_BIN + os.pathsep + os.environ.get("PATH", "")
    _icds = [p for p in os.environ.get("OCL_ICD_FILENAMES", "").split(os.pathsep) if p]
    if _INTEL_DLL not in _icds:
        os.environ["OCL_ICD_FILENAMES"] = os.pathsep.join(_icds + [_INTEL_DLL])

from pyopencl.tools import get_gl_sharing_context_properties
import pyopencl as cl
import warnings
warnings.filterwarnings("ignore", category=cl.RepeatedKernelRetrieval)

_works_cache = {}

def _device_works(device):
    """True if a trivial kernel can be built AND executed on the device. A runtime can be listed
    but unusable (e.g. PoCL on Windows without a working linker : kernels compile, but never run)."""
    key = (device.platform.name, device.name)
    if key not in _works_cache:
        try:
            import numpy as np
            ctx = cl.Context([device])
            queue = cl.CommandQueue(ctx)
            prg = cl.Program(ctx, "__kernel void t(__global int *a){ a[get_global_id(0)] = 42; }").build()
            out = np.zeros(8, dtype=np.int32)
            buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out.nbytes)
            prg.t(queue, (8,), None, buf)
            cl.enqueue_copy(queue, out, buf).wait()
            _works_cache[key] = bool((out == 42).all())
        except Exception:
            _works_cache[key] = False
    return _works_cache[key]


def list_devices():
    """List every usable OpenCL device : [{id, platform, name, type, compute_units, mem_mb}].
    `id` is the index in this list, usable as OCL_Driver(device_index=id).
    Devices that cannot run a kernel are left out."""
    devices = []
    for platform in cl.get_platforms():
        try:
            platform_devices = platform.get_devices()
        except cl.Error:
            continue
        for device in platform_devices:
            if not _device_works(device):
                continue
            devices.append({
                "id": len(devices),
                "platform": platform.name.strip(),
                "name": device.name.strip(),
                "type": "GPU" if device.type & cl.device_type.GPU else "CPU" if device.type & cl.device_type.CPU else "OTHER",
                "compute_units": device.max_compute_units,
                "mem_mb": device.global_mem_size // 1024 // 1024,
            })
    return devices


class OCL_Driver:
    """Context, queue and compiled kernels for one OpenCL device.

    workers_exp1 / workers_exp2 : work-group size is (2**exp1, 2**exp2) = 32 x 32 by default ; the grid
    sizes must be multiples of it (None lets OpenCL choose). Attributes : ctx, queue, prg (the kernels,
    called as prg.<kernel_name>(queue, global_size, local_size, args...)), local_size_2D, local_size_inlet.
    """
    def __init__(self,  dev_id=0, kernel_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.cl"), 
                 workers_exp1 = 5, workers_exp2 = 5, interop=False, device_index=None):
        """dev_id : index of the OpenCL platform (first device of the platform is used).
        device_index : if given, index in list_devices() (overrides dev_id)."""
        
        if device_index is not None:
            all_devices = [(p, d) for p in cl.get_platforms() for d in p.get_devices() if _device_works(d)]
            self.platform, self.device = all_devices[device_index]
        else:
            self.platform = cl.get_platforms()[dev_id]    # Select the  platform [dev_id]
            self.device = self.platform.get_devices()[0]  # Select the first device on this platform [0]
        print('You ve selected the plateform : %s'%(self.platform.name))
                
        interop = interop and cl.have_gl()   # pyopencl wheels are often built without GL sharing
        self.interop = interop
        if(interop):        
            if sys.platform == "darwin":
                self.ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                    devices=[])
            else:
                try:
                    self.ctx = cl.Context(properties=[
                        (cl.context_properties.PLATFORM, self.platform)]
                        + get_gl_sharing_context_properties())
                except:
                    self.ctx = cl.Context(properties=[
                        (cl.context_properties.PLATFORM, self.platform)]
                        + get_gl_sharing_context_properties(),
                        devices = [self.platform.get_devices()[0]])
        else:
            self.ctx = cl.Context([self.device])  
              
        # Create kernels
        with open(kernel_file, encoding="utf-8") as kernel_source:
            self.prg = cl.Program(self.ctx, kernel_source.read()).build()
          
        # Create a simple queue
        self.queue = cl.CommandQueue(self.ctx, self.ctx.devices[0],
                                     properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        if (workers_exp1==None or workers_exp2==None):            
            self.local_size_2D = None
            self.local_size_3D = None
            self.local_size_inlet = None
        else :
            val1 = 2**workers_exp1
            val2 = 2**workers_exp2
            self.local_size_2D = (val1,val2)
            self.local_size_3D = (1,val1,val2)
            self.local_size_inlet = (val2,)
            
    def describe(self):
        """Short text telling where the solver runs, e.g. 'GPU : NVIDIA GeForce RTX 4080 (76 compute units)'
        or 'CPU : <name> sur 16 threads'."""
        name = self.device.name.strip()
        if self.device.type & cl.device_type.GPU:
            return "GPU : %s (%d compute units)" % (name, self.device.max_compute_units)
        if self.device.type & cl.device_type.CPU:
            return "CPU : %s sur %d threads" % (name, self.device.max_compute_units)
        return "%s (%d compute units)" % (name, self.device.max_compute_units)

    def print_spec(self):
        print("===============================================================")
        print("Platform name:", self.platform.name)
        print("Platform profile:", self.platform.profile)
        print("Platform vendor:", self.platform.vendor)
        print("Platform version:", self.platform.version)
        print("---------------------------------------------------------------")
        print("Device name:", self.device.name)
        print("Device type:", cl.device_type.to_string(self.device.type))
        print("Device memory: ", self.device.global_mem_size//1024//1024, 'MB')
        print("Device max clock speed:", self.device.max_clock_frequency, 'MHz')
        print("Device compute units:", self.device.max_compute_units)
        print("Device max work group size:", self.device.max_work_group_size)
        print("Device max work item sizes:", self.device.max_work_item_sizes)