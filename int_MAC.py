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
  const int i = (blockDim.x * blockIdx.x + threadIdx.x ) * 20 ;
  int j = 0;
  int a1 = a[i];
  int b1 = b[i];
  int d1 = 0;
  int a2 = a[i+1];
  int b2 = b[i+1];
  int d2 = 0;
  int a3 = a[i+2];
  int b3 = b[i+2];
  int d3 = 0;
  int a4 = a[i+3];
  int b4 = b[i+3];
  int d4 = 0;
  int a5 = a[i+4];
  int b5 = b[i+4];
  int d5 = 0;
  int a6 = a[i+5];
  int b6 = b[i+5];
  int d6 = 0;
  int a7 = a[i+6];
  int b7 = b[i+6];
  int d7 = 0;
  int a8 = a[i+7];
  int b8 = b[i+7];
  int d8 = 0;
  int a9 = a[i+8];
  int b9 = b[i+8];
  int d9 = 0;
  int a10 = a[i+9];
  int b10 = b[i+9];
  int d10 = 0;

  int a11 = a[i+10];
  int b11 = b[i+10];
  int d11 = 0;
  int a12 = a[i+11];
  int b12 = b[i+11];
  int d12 = 0;
  int a13 = a[i+12];
  int b13 = b[i+12];
  int d13 = 0;
  int a14 = a[i+13];
  int b14 = b[i+13];
  int d14 = 0;
  int a15 = a[i+14];
  int b15 = b[i+14];
  int d15 = 0;
  int a16 = a[i+15];
  int b16 = b[i+15];
  int d16 = 0;
  int a17 = a[i+16];
  int b17 = b[i+16];
  int d17 = 0;
  int a18 = a[i+17];
  int b18 = b[i+17];
  int d18 = 0;
  int a19 = a[i+18];
  int b19 = b[i+18];
  int d19 = 0;
  int a20 = a[i+19];
  int b20 = b[i+19];
  int d20 = 0;

  int cnt = count;
  for(j = 0; j < cnt ; j++)
  {
  	d1 += a1 * b1;
  	d2 += a2 * b2;
  	d3 += a3 * b3;
  	d4 += a4 * b4;
  	d5 += a5 * b5;
  	d6 += a6 * b6;
  	d7 += a7 * b7;
  	d8 += a8 * b8;
  	d9 += a9 * b9;
  	d10 += a10 * b10;
  	d11 += a11 * b11;
  	d12 += a12 * b12;
  	d13 += a13 * b13;
  	d14 += a14 * b14;
  	d15 += a15 * b15;
  	d16 += a16 * b16;
  	d17 += a17 * b17;
  	d18 += a18 * b18;
  	d19 += a19 * b19;
  	d20 += a20 * b20;
  }
  dest[i] = d1;
  dest[i+1] = d2;
  dest[i+2] = d3;
  dest[i+3] = d4;
  dest[i+4] = d5;
  dest[i+5] = d6;
  dest[i+6] = d7;
  dest[i+7] = d8;
  dest[i+8] = d9;
  dest[i+9] = d10;
  dest[i+10] = d11;
  dest[i+11] = d12;
  dest[i+12] = d13;
  dest[i+13] = d14;
  dest[i+14] = d15;
  dest[i+15] = d16;
  dest[i+16] = d17;
  dest[i+17] = d18;
  dest[i+18] = d19;
  dest[i+19] = d20;
}
""")

multiply_them = mod.get_function("multiply_them")

threadsPerBlock = 1000
numBlocks = 2000
count = 100000
a = np.random.randn(threadsPerBlock * numBlocks * 20).astype(np.int32)
b = np.random.randn(threadsPerBlock * numBlocks * 20).astype(np.int32)
c = np.random.randn(threadsPerBlock * numBlocks * 20).astype(np.int32)
dest = np.zeros_like(a)
for i in range (100):
	multiply_them(
	        drv.Out(dest), drv.In(a), drv.In(b), np.int32(count),
	        block=(threadsPerBlock,1,1), grid=(numBlocks,1,1))

print(dest-a*b)
