//
// Created by Artyom on 4/9/2023.
//

#ifndef CUDAMATRIXCALCULATOR_GAUS_CUH
#define CUDAMATRIXCALCULATOR_GAUS_CUH
template <typename T>
__global__ void kernel_SLAE_To_Triangle(T *A, T *B, int n, int number_row,
                                        int number_column) {
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;

  unsigned int row = by * TILE_DIM + ty;
  unsigned int column = bx * TILE_DIM + tx;

  if (number_column > column || number_row >= row || row >= n)
    return;

  T coefficient = A[row * n + number_column] / A[number_row * n + number_column];

  if (column < n) {
      A[row * n + column] -= coefficient * A[number_row * n + column];
  }
  if(column == n){
    B[row] -= coefficient*B[number_row];
  }
}

template <typename T>
__global__ void kernel_down(T *A, int n, int number_row, int number_column) {
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;

  unsigned int row = by * TILE_DIM + ty;
  unsigned int column = bx * TILE_DIM + tx;

  if (number_column > column || number_row >= row || row >= n || column >= n)
    return;
  T coefficient =A[row * n + number_column] / A[number_row * n + number_column];
  if (number_column == column) {
    A[row * n + column] = 0;
  } else {
    A[row * n + column] -= coefficient * A[number_row * n + column];
  }
}

template <typename T>
__global__ void kernel_up(T *A, T *B, int n) {
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;

  unsigned int row = by * TILE_DIM + ty;
  unsigned int column = bx * TILE_DIM + tx;

  if(row > n)
    return;
  if(column==0){
    B[row] /= A[row * n + row];
  }
  if(column < n){
    A[row * n + column] /= A[row * n + row];
  }
}
#endif // CUDAMATRIXCALCULATOR_GAUS_CUH
