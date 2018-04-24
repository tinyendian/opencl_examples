import pyopencl as cl
import pyopencl.array as cla
import pyopencl.clmath as clm
import numpy as np
import os
from timeit import default_timer as timer

# Create NumPy arrays and fill them with random numbers
# Use single precision, not all OpenCL devices support double precision
class data:
    def __init__(self, rank, seed):
        self.rank = rank
        np.random.seed(seed)
        self.a = np.random.rand(rank, rank).astype(np.float32)
        self.b = np.random.rand(rank, rank).astype(np.float32)

# Configure OpenCL context and queue
class opencl_setup:
    def __init__(self, platform, device):

        # Switch on compiler output for debugging
        os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"

        platforms = cl.get_platforms()
        self.plat = platforms[platform]
        devices = self.plat.get_devices()
        self.dev = devices[device]

        print("===============================================================")
        print("Setting OpenCL platform ID %i of %i platforms" % (platform, len(platforms)))
        print("Platform name: %s" % self.plat.name)
        print("Platform profile: %s" % self.plat.profile)
        print("Platform vendor: %s" % self.plat.vendor)
        print("Platform version: %s" % self.plat.version)
        print("---------------------------------------------------------------")
        print("Setting device ID %i of %i devices" % (platform, len(devices)))
        print("Dvice name: %s" % self.dev.name)
        print("Device type: %s" % cl.device_type.to_string(self.dev.type))
        print("Device memory: %f MB" % (self.dev.global_mem_size/(1024**2)))
        print("Device max clock speed: %f MHz" % self.dev.max_clock_frequency)
        print("Device compute units: %s" % self.dev.max_compute_units)
        print("Device max work group size: %s" % self.dev.max_work_group_size)
        print("Device max work item sizes: %s" % self.dev.max_work_item_sizes)

        # Create OpenCL context and command queue
        # Note that context must be kept in scope, so it has to become
        # a class property
        self.ctx = cl.Context(devices)
        self.queue = cl.CommandQueue(self.ctx, device=self.dev, properties=cl.command_queue_properties.PROFILING_ENABLE)

        self.mf = cl.mem_flags

#
# NumPy version
#

class np_matmul_bm(data):
    def __init__(self, rank, seed, niter):
        data.__init__(self, rank, seed)
        self.niter = niter
        self.mintime = None
        self.perf = None
        self.maxreldev = None
        self.c = np.zeros(self.a.shape, dtype=np.float32)

    def compute(self):
        self.mintime = 1.0e9
        for iter in range(0, self.niter):
            start = timer()
            self.c = np.matmul(self.a, self.b)
            end = timer()
            self.mintime = np.min([end-start, self.mintime])
            self.perf = 2.0e-9*self.rank**3/self.mintime

    def check(self, ref_result):
        # The NumPy result is the reference result
        self.maxreldev = 0.0

    def show(self):
        print("Best NumPy time: %e s" % self.mintime)
        print("Best NumPy perf: %f GFLOPS" % self.perf)
        print("Maximum relative deviation: %e" % self.maxreldev)

    def get_result(self):
        return self.c

#
# OpenCL version
#

class cl_matmul_bm(opencl_setup, data):
    def __init__(self, platform, device, rank, seed, niter):
        opencl_setup.__init__(self, platform, device)
        data.__init__(self, rank, seed)
        self.niter = niter
        self.mintime = None
        self.perf = None
        self.maxreldev = None
        self.c = np.zeros(self.a.shape, dtype=np.float32)

        self.prg = cl.Program(self.ctx, """
    __kernel void matmul(const __global float *a, const __global float *b,
                     __global float *c, ushort const n)
    {
      const int g_row = get_global_id(0);
      const int g_col = get_global_id(1);
      float sum = 0.0;
      if ( (g_row<n)  && (g_col<n)) {
        for(int idx = 0; idx < n; idx++) {
          sum += a[g_col*n + idx] * b[idx*n + g_row];
        }
        c[g_col*n + g_row] = sum;
      }
    }
        """).build()


    def compute(self):
        a_opencl = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.a)
        b_opencl = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.b)
        c_opencl = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.c.nbytes)
        self.mintime = 1.0e9
        for iter in range(0, self.niter):
            run_kernel = self.prg.matmul(self.queue, self.c.shape, None, a_opencl, b_opencl,
                                    c_opencl, np.uint16(self.rank))
            run_kernel.wait()
            self.mintime = np.min([1e-9*(run_kernel.profile.end - run_kernel.profile.start), self.mintime])
        self.perf = 2.0e-9*self.rank**3/self.mintime
        cl.enqueue_copy(self.queue, self.c, c_opencl).wait()

    def check(self, ref_result):
        self.maxreldev = np.max(np.abs(self.c - ref_result)/ref_result)

    def show(self):
        print("Best OpenCL time: %e s" % self.mintime)
        print("Best OpenCL perf: %f GFLOPS" % self.perf)
        print("Maximum relative deviation: %e" % self.maxreldev)

    def get_result(self):
        return self.c

def main():

    # Benchmark parameters
    rank = 1000
    niter = 10
    random_seed = 12345
    cl_platform = 0
    cl_device = 0

    np_test = np_matmul_bm(rank, random_seed, niter)
    np_test.compute()
    np_test.check(None)
    np_test.show()

    cl_test = cl_matmul_bm(cl_platform, cl_device, rank, random_seed, niter)
    cl_test.compute()
    cl_test.check(np_test.get_result())
    cl_test.show()
  
if __name__== "__main__":
    main()