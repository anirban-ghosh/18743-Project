from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b, int count)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = 0;
  float al = a[i];
  float bl = b[i];
  float dl;
  int cnt = count;
  for(j = 0; j < 100000 ; j++)
  {
  	dl += al * bl;
  }
  dest[i] = dl;
}
""")

multiply_them = mod.get_function("multiply_them")

threadsPerBlock = 1000
numBlocks = 20000
count = 10000
a = np.random.randn(threadsPerBlock * numBlocks).astype(np.float32)
b = np.random.randn(threadsPerBlock * numBlocks).astype(np.float32)
c = np.random.randn(threadsPerBlock * numBlocks).astype(np.float32)
dest = np.zeros_like(a)
for i in range (100):
	multiply_them(
	        drv.Out(dest), drv.In(a), drv.In(b), np.int32(count),
	        block=(threadsPerBlock,1,1), grid=(numBlocks,1,1))

print(dest-a*b)
