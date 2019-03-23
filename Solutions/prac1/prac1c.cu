//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


//
// kernel routine
//

__global__ void my_first_kernel(float *x, float *y)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  //x[tid] = (float) threadIdx.x;
  x[tid] = x[tid] + y[tid];
}


//
// main code
//

int main(int argc, const char **argv)
{
  float *x, *y;
  int   nblocks, nthreads, nsize, n;

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  checkCudaErrors(cudaMallocManaged(&x, nsize*sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&y, nsize*sizeof(float)));

  x = (float *)malloc(nsize*sizeof(float));
  y = (float *)malloc(nsize*sizeof(float));
  for (n=0; n<nsize; n++){
    x[n] = n;
    y[n] = 100-n;
  }

  // execute kernel

  my_first_kernel<<<nblocks,nthreads>>>(x,y);
  getLastCudaError("my_first_kernel execution failed\n");

  // synchronize to wait for kernel to finish, and data copied back

  cudaDeviceSynchronize();

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,x[n]);

  // free memory

  checkCudaErrors(cudaFree(x));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
