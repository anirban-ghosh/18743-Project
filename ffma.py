from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void multiply_them(int *dest, int *a, int *b, int count)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = 0;
  int al = a[i];
  int bl = b[i];
  int al1 = al+1;
  int bl1 = bl+2;
  int dl1 = 0;
  int dl2 = 0;
  int cnt = count;
  int k = 0;
  for(j = 0; j < cnt ; j++)
  {
  	dl1 += al * bl;
  	dl2 = al1 * bl1 + dl1;
  }
  dest[i] = dl1 + dl2;
}
""")

multiply_them = mod.get_function("multiply_them")

threadsPerBlock = 1000
numBlocks = 2000
count = 10000
a = np.random.randn(threadsPerBlock * numBlocks).astype(np.int32)
b = np.random.randn(threadsPerBlock * numBlocks).astype(np.int32)
c = np.random.randn(threadsPerBlock * numBlocks).astype(np.int32)
dest = np.zeros_like(a)
for i in range (100):
	multiply_them(
	        drv.Out(dest), drv.In(a), drv.In(b), np.int32(count),
	        block=(threadsPerBlock,1,1), grid=(numBlocks,1,1))

print(dest-a*b)
