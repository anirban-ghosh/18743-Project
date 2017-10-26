from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void idpTest(int * __restrict__ dest, int * __restrict__ a, int * __restrict__ b, int * __restrict__ c, int count)
{
  const int i = (blockDim.x * blockIdx.x + threadIdx.x);
  int j;
  int d = 0;
  for (j = 0; j < count; j++)
  {
  	d = __dp4a(a[i], b[i], c[i]);
  	//dest[i] = __dp2a_lo(a[i], b[i], j);
  	//dest[i] = a[i]*b[i] + j;
  }
  dest[i] = d;
}
""")

idpTest = mod.get_function("idpTest")

threadsPerBlock = 1000
numBlocks = 10000
# exit()
count = 100000
a = np.random.randn(threadsPerBlock * numBlocks).astype(np.int32)
b = np.random.randn(threadsPerBlock * numBlocks).astype(np.int32)
c = np.random.randn(threadsPerBlock * numBlocks).astype(np.int32)
dest = np.zeros_like(a)
for i in range (10):
	idpTest(
	        drv.Out(dest), drv.In(a), drv.In(b), drv.In(c), np.int32(count),
	        block=(threadsPerBlock,1,1), grid=(numBlocks,1,1))

# print(dest-a*b)
