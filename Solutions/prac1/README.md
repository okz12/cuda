### Main Learnings:

1. Use of `checkCudaErrors` makes code more robust. Initialising with 0 blocks/threads would not give an output normally but would be caught by the function.

2. Can `printf` statements but do not follow a necessary order.

3. Syntax (prac1b)-- `findCudaDevice`, `cudaMalloc`, `cudaMemcpy` (bidirectional), `cudaFree`, `cudaDeviceReset`

4. Use of `cudaMallocManaged` reduces need of `cudaMalloc`, `cudaMemcpy` H->D (copied when kernel is started) and `cudaMemcpy` D->H (replaced with `cudaDeviceSynchronize`). No need to handle GPU pointers.
