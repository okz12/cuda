
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

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ float a, b, c;
__constant__ int NUM;

////////////////////////////////////////////////////////////////////////
// kernel for averaging az^2 + bz + c
////////////////////////////////////////////////////////////////////////
__global__ void avg_eq(float *x)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  curandState randState;
  curand_init(seed, tid, 0, &randState);
  int randInteger = curand(&randState);
  float z = curand_normal(&randState);
  x[tid] = (a*a*z + b*z + c)/NUM;

  //sequential addressing add
}

__global__ void reduce(float *x, float *o)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  int s;
  for (s=1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      x[index] += x[index + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0){
    o[blockIdx.x] = x[tid];
    printf("%f\n",x[tid]);
  }
  printf("gridDim: %d %d, threadIdx: %d %d, blockIdx: %d %d, blockDim: %d %d, x: %f, o: %f, %d\n",
  gridDim.x, gridDim.y, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, x[tid], o[tid], s);
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

  float   h_a, h_b, h_c;
  int     h_NUM;
  float   *h_x, *h_o, *d_x, *d_o;
  h_NUM = 256;
  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_x = (float *)malloc(sizeof(float)*h_NUM);
  h_o = (float *)malloc(sizeof(float)*h_NUM);

  checkCudaErrors( cudaMalloc((void **)&d_x, sizeof(float)*h_NUM) );
  checkCudaErrors( cudaMalloc((void **)&d_o, sizeof(float)*h_NUM) );

  h_a = 5.0f;
  h_b = 10.0f;
  h_c = 15.0f;

  checkCudaErrors( cudaMemcpyToSymbol(a,    &h_a,    sizeof(h_a)) );
  checkCudaErrors( cudaMemcpyToSymbol(b,    &h_b,    sizeof(h_b)) );
  checkCudaErrors( cudaMemcpyToSymbol(c,    &h_c,    sizeof(h_c)) );
  checkCudaErrors( cudaMemcpyToSymbol(NUM,    &h_NUM,    sizeof(h_NUM)) );

  // execute kernel and time it

  cudaEventRecord(start);

  avg_eq<<<h_NUM/64, 64>>>(d_x);
  getLastCudaError("avg_eq execution failed\n");
  reduce<<<h_NUM/64, 64>>>(d_x, d_o);
  getLastCudaError("reduce execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("Monte Carlo kernel execution time (ms): %f \n",milli);

  // copy back results

  checkCudaErrors( cudaMemcpy(h_x, d_x, sizeof(float)*h_NUM,
                   cudaMemcpyDeviceToHost) );
 checkCudaErrors( cudaMemcpy(h_o, d_o, sizeof(float)*h_NUM,
                  cudaMemcpyDeviceToHost) );

  // compute average

  float sum1 = 0.0;
  for (int i=0; i<h_NUM; i++) {
    sum1 += h_x[i];
  }

  float sum2 = 0.0;
  for (int i=0; i<h_NUM; i++) {
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
