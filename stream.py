from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void streamPerf(float * __restrict__ const dest, float const * __restrict__ const a)
{
  const int i = (blockDim.x * blockIdx.x + threadIdx.x);
  dest[i] = a[i];
}
""")

streamPerf = mod.get_function("streamPerf")

threadsPerBlock = 1024
numBlocks = (1024*256)*3
wordSize = 4
print ("Size = ", threadsPerBlock*numBlocks*wordSize/(1024**3), "GB")
# exit()
# count = 1000000
a = np.random.randn(threadsPerBlock * numBlocks).astype(np.float32)
dest = np.zeros_like(a)
for i in range (10):
	streamPerf(
	        drv.Out(dest), drv.In(a), 
	        block=(threadsPerBlock,1,1), grid=(numBlocks,1,1))

# print(dest-a*b)
