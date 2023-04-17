//
// Created by Artyom on 4/17/2023.
//

#ifndef CUDAMATRIXCALCULATOR_PLUFACTORISATION_CUH
#define CUDAMATRIXCALCULATOR_PLUFACTORISATION_CUH

__shared__ unsigned int sidx[TILE_DIM];

template <typename T>
__global__ void kernel_swap_rows_and_simplify(T *U, T *P, T *L, int n, int k, int width, int height) {

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;

  unsigned int row = by * TILE_DIM + ty;
  unsigned int column = bx * TILE_DIM + tx;

  unsigned int i = row*TILE_DIM+column;

  if(column < width) {
    T tmp = U[n * width + column];
    U[n * width + column] = U[k * width + column];
    U[k * width + column] = tmp;

    tmp = P[n * width + column];
    P[n * width + column] = P[k * width + column];
    P[k * width + column] = tmp;

    if(column < n){
      tmp = L[n * width + column];
      L[n * width + column] = L[k * width + column];
      L[k * width + column] = tmp;
    }
  }
  __syncthreads();

  T coefficient = U[row * width + n] / U[n * width + n];
  if(column < width && row > n && row < height){
    U[row*width+column] -= coefficient * U[n*width+column];
    if(column == n)
    L[row*width+column] = coefficient * L[n*width+column];
  }
//  if (column < n) {
//    U[row * n + column] -= coefficient * U[number_row * n + column];
//  }
//  if(column == n){
//    B[row] -= coefficient*B[number_row];
//  }

}

template <typename T>
__global__ void kernel_max_mod_cols(T *a, T *d, unsigned int* maxIdx, int col, int width, int height)
{
  __shared__ T sdata[TILE_DIM]; //"static" shared memory

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;

  unsigned int row = by * TILE_DIM + ty;
  unsigned int column = bx * TILE_DIM + tx;

  unsigned int i = row*TILE_DIM+column;

  if(i < height && i >= col) {
    sdata[tx] = a[i * width + col];
    sidx[tx] = i;
  }else {
    sdata[tx] = std::numeric_limits<T>::min();
    sidx[tx] = i;
  }
  __syncthreads();
  for(unsigned int s=TILE_DIM/2 ; s >= 1 ; s=s/2)
  {
    if(tx < s)
    {
      if(abs(sdata[tx]) < abs(sdata[tx + s]))
      {
        sdata[tx] = sdata[tx + s];
        sidx[tx] = sidx[tx+s];
      }
    }
    __syncthreads();
  }
  if(tx == 0 )
  {
    d[blockIdx.x] = sdata[0];
    //printf("%f\n", sdata[0]);
    maxIdx[blockIdx.x] = sidx[0];
  }
}



#endif // CUDAMATRIXCALCULATOR_PLUFACTORISATION_CUH
