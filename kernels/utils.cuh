//
// Created by Artyom on 4/9/2023.
//

#ifndef CUDAMATRIXCALCULATOR_UTILS_CUH
#define CUDAMATRIXCALCULATOR_UTILS_CUH

#define gpuErrorChek(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUAssert: %s %d %s %d\n", cudaGetErrorString(code), code, file, line);
    if (abort) exit(code);
  }
}

#endif // CUDAMATRIXCALCULATOR_UTILS_CUH
