# -*- coding: utf-8 -*-
from pyopencl.tools import get_gl_sharing_context_properties
import pyopencl as cl
import sys

class OCL_Driver:
    def __init__(self,  dev_id, kernel_file="kernels.cl", 
                 workers_exp1 = 5, workers_exp2 = 5, interop=False):
        
        self.platform = cl.get_platforms()[dev_id]    # Select the  platform [dev_id]
        print('You ve selected the plateform : %s'%(self.platform.name))
        
        # Create a context with all the devices
        self.device = self.platform.get_devices()[0]  # Select the first device on this platform [0]
                
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
        self.prg = cl.Program(self.ctx, open(kernel_file).read()).build()
          
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