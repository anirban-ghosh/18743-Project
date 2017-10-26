from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule

kernel_code_template="""
#include "cuda_fp16.h" 
__global__ void multiply_them(float * __restrict__ dest, float * __restrict__ a, float * __restrict__ b, int count)
{
  const int i = (blockDim.x * blockIdx.x + threadIdx.x );
  int j = 0;
  __half2 a1 = __float2half2_rn(a[i]);
  __half2 b1 = __float2half2_rn(b[i]);
  __half2 d1 = __float2half2_rn(0.0);
  int cnt = count;
  float2 temp;
  for(j = 0; j < cnt ; j++)
  {
  	d1 = __hfma2(a1 ,b1,d1) ;
  }
  temp = __half22float2(d1);
  dest[i] = temp.x;
}
"""

mod = SourceModule(kernel_code_template)
multiply_them = mod.get_function("multiply_them")

threadsPerBlock = 1000
numBlocks = 2000
count = 1000
a = np.random.randn(threadsPerBlock * numBlocks  ).astype(np.float32)
b = np.random.randn(threadsPerBlock * numBlocks  ).astype(np.float32)
c = np.random.randn(threadsPerBlock * numBlocks  ).astype(np.float32)
dest = np.zeros_like(a)
for i in range (100):
	multiply_them(
	        drv.Out(dest), drv.In(a), drv.In(b), np.int32(count),
	        block=(threadsPerBlock,1,1), grid=(numBlocks,1,1))

print(dest-a*b)
