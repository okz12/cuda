### Main Learnings

1. Timing Code:

  `cudaEventRecord(start);

  pathcalc<<<NPATH/64, 64>>>(d_z, d_v);
  getLastCudaError("pathcalc execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);`

2. Use of stratified / distanced memory hurts performance. Latency for memories: Register < Local < Shared < Global. Version 2 is slower (~20ms) than version 1 (~4ms)
