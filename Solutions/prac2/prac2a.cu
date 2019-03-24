
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
//#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#define seed 123
#define BLOCKSIZE 64

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ float a, b, c;
__constant__ int NUM;

////////////////////////////////////////////////////////////////////////
// kernel for averaging az^2 + bz + c
////////////////////////////////////////////////////////////////////////
__global__ void avg_eq(float *x, float *o)
{
  float total = 0;
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  curandState randState;
  curand_init(seed, tid, 0, &randState);
  int randInteger = curand(&randState);
  float z = curand_normal(&randState);
  x[tid] = (a*a*z + b*z + c)/NUM;

  __syncthreads();

  if (threadIdx.x == 0)
  {
    for (int s=tid; s<tid + blockDim.x; s++)
    {
      total += x[s];
    }
    o[blockIdx.x] = total;
    //printf("%d %d: %f\n", tid, tid+blockDim.x, total);
  }
  else
  {
    o[tid] = 0;
  }
}

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

  float   h_a, h_b, h_c;
  int     h_NUM;
  float   *h_x, *h_o, *d_x, *d_o;
  h_NUM = 64;
  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_x = (float *)malloc(sizeof(float)*h_NUM);
  h_o = (float *)malloc(sizeof(float)*h_NUM/BLOCKSIZE);

  checkCudaErrors( cudaMalloc((void **)&d_x, sizeof(float)*h_NUM) );
  checkCudaErrors( cudaMalloc((void **)&d_o, sizeof(float)*h_NUM/BLOCKSIZE) );

  h_a = 5.0f;
  h_b = 10.0f;
  h_c = 15.0f;

  checkCudaErrors( cudaMemcpyToSymbol(a,    &h_a,    sizeof(h_a)) );
  checkCudaErrors( cudaMemcpyToSymbol(b,    &h_b,    sizeof(h_b)) );
  checkCudaErrors( cudaMemcpyToSymbol(c,    &h_c,    sizeof(h_c)) );
  checkCudaErrors( cudaMemcpyToSymbol(NUM,    &h_NUM,    sizeof(h_NUM)) );

  // execute kernel and time it

  cudaEventRecord(start);

  avg_eq<<<h_NUM/64, 64>>>(d_x, d_o);
  getLastCudaError("avg_eq execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("Monte Carlo kernel execution time (ms): %f \n",milli);

  // copy back results

  checkCudaErrors( cudaMemcpy(h_x, d_x, sizeof(float)*h_NUM,
                   cudaMemcpyDeviceToHost) );
 checkCudaErrors( cudaMemcpy(h_o, d_o, sizeof(float)*h_NUM/BLOCKSIZE,
                  cudaMemcpyDeviceToHost) );

  // compute average

  float sum1 = 0.0;
  for (int i=0; i<h_NUM; i++) {
    sum1 += h_x[i];
  }

  float sum2 = 0.0;
  for (int i=0; i<h_NUM/BLOCKSIZE; i++) {
    sum2 += h_o[i];
  }

  printf("Average: %f %f\n", sum1, sum2);

  // Release memory and exit cleanly

  free(h_x);
  checkCudaErrors( cudaFree(d_x) );
  free(h_o);
  checkCudaErrors( cudaFree(d_o) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

}
