//
// Created by artyom on 9/8/21.
//

#include "Matrix.cuh"

#define gpuErrorChek(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUAssert: %s %d %s %d\n", cudaGetErrorString(code), code, file, line);
        if (abort) exit(code);
    }
}

__global__ void addMatrix(const float* A, const float* B, float* C, int Cols){
    unsigned int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    unsigned int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    C[Row*Cols+Col] = A[Row*Cols+Col] + B[Row*Cols+Col];
}

__global__ void
MatMul(const float* A,const float* B, float* C, int ARows, int ACols, int BRows,
       int BCols, int CRows, int CCols)
{
    float CValue = 0;

    unsigned int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    unsigned int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

        if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
            As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
            Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int n = 0; n < TILE_DIM; ++n)
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

        __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
          (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

template<typename T>
__global__ void kernel_SLAE_To_Triangle(T *A, T *B,int n,int number_row,int number_column){
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;

    unsigned int tx=threadIdx.x;
    unsigned int ty=threadIdx.y;

    unsigned int row=by*TILE_DIM+ty;
    unsigned int column=bx*TILE_DIM+tx;

    if(number_column<=column && number_row<row && row<n && column<n){
        T coefficient =A[row*n+number_column]/A[number_row*n+number_column];
        //printf("%d %d %f %f %f %f %f\n", row, column, A[row*n+number_column], A[number_row*n+number_column], coefficient, A[row*n+column], A[number_row*n+column]);
        if(number_column==column) {
            //B[row]-=B[number_row]*coefficient;
            A[row*n+column] = 0;
        }else{
            A[row*n+column]-=coefficient*A[number_row*n+column];
        }
    }
}

template<typename T>
__global__ void kernel_down(T *A, int n,int number_row,int number_column){
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;

    unsigned int tx=threadIdx.x;
    unsigned int ty=threadIdx.y;

    unsigned int row=by*TILE_DIM+ty;
    unsigned int column=bx*TILE_DIM+tx;

    if(number_column<=column && number_row<row && row<n && column<n){
        T coefficient =A[row*n+number_column]/A[number_row*n+number_column];
        //printf("%d %d %d %d %f %f %f %f %f\n", row, number_row, column, number_column, A[row*n+number_column], A[number_row*n+number_column], coefficient, A[row*n+column], A[number_row*n+column]);
        if(number_column==column) {
            //B[row]-=B[number_row]*coefficient;
            A[row*n+column] = 0;
        }else{
            A[row*n+column]-=coefficient*A[number_row*n+column];
        }
    }
}

template<typename T>
__attribute__((unused))
__global__ void kernel_up(T *A,T *B,int n,int number_row,int number_column){
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;

    unsigned int tx=threadIdx.x;
    unsigned int ty=threadIdx.y;

    unsigned int row=by*TILE_DIM+ty;
    unsigned int column=bx*TILE_DIM+tx;

    if(number_column==column && number_row>row){
        T coefficient=A[row*n+number_column]/A[number_row*n+number_column];

        B[row]-=B[number_row]*coefficient;
        A[row*n+column]=0;
    }
}



template<typename T>
Matrix<T> Matrix<T>::to_triangle(){
    T *dev_a;

    cudaMalloc((void **) &dev_a, this->width * this->height * sizeof(T ));

    cudaMemcpy(dev_a, this->matrix, this->width * this->height  * sizeof(T), cudaMemcpyHostToDevice);

    dim3 Grid(ceil(this->width/(TILE_DIM*1.)), ceil(this->height/(TILE_DIM*1.)));
    dim3 Block(TILE_DIM, TILE_DIM);

    cudaEvent_t start, stop;
    float gpuTime = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, nullptr);

    for (int i = 0; i < this->height - 1; i++) {
        kernel_down<<<Grid, Block>>>(dev_a, this->height, i, i);
        gpuErrorChek(cudaPeekAtLastError())
        gpuErrorChek(cudaDeviceSynchronize())
    }
    //for (int i = this->height - 1; i >= 0; i--) kernel_up<<<Grid,Block>>>(dev_a, dev_b, this->height, i, i);

    Matrix<T> result(this->width, this->height);

    cudaMemcpy(result.data(),dev_a,this->width * this->height*sizeof(T),cudaMemcpyDeviceToHost);

    //for (int i = 0; i < n - 1; i++) h_C[i]=h_B[i]/h_A[i*n+i];

    cudaEventRecord(stop,nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime,start,stop);

//    this->print();

    cudaFree(dev_a);
    cudaDeviceReset();

    return result;
}

template<typename T>
Matrix<T>::Matrix(int width, int height, T arg, ...) {
    this->width = width;
    this->height = height;
    va_list args;
    va_start(args, arg);
    this->matrix=(T *) malloc(width * height * sizeof(T));
    matrix[0] = arg;
    for(int i = 0; i<height;i++){
        for(int j = i==0?1:0; j<width;j++){
            matrix[i*width+j] = va_arg(args, T);
        }
    }
    va_end(args);
}

template<typename T>
Matrix<T>::Matrix(int width, int height) {
    this->width = width;
    this->height = height;
    this->matrix=(T *) malloc(width * height * sizeof(T));
}

template<typename T>
Matrix<T> Matrix<T>::operator*(Matrix<T> m1) {

    int CWidth = this->width;
    int CHeight = m1.getHeight();
    Matrix result = Matrix(CWidth, CHeight);

    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid;

    dimGrid.x = (CWidth + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (CHeight + dimBlock.y - 1) / dimBlock.y;

    T *deviceA, *deviceB, *deviceC;

    struct timeval t1{}, t2{};

    gettimeofday(&t1, nullptr);


    gpuErrorChek(cudaMalloc((void **) &deviceA, this->width * this->height * sizeof(T)))
    gpuErrorChek(cudaMalloc((void **) &deviceB, m1.getWidth() * m1.getHeight() * sizeof(T)))
    gpuErrorChek(cudaMalloc((void **) &deviceC, CWidth * CHeight * sizeof(T)))

    gpuErrorChek(cudaMemcpy(deviceA, this->data(), this->width * this->height * sizeof(T), cudaMemcpyHostToDevice))
    gpuErrorChek(cudaMemcpy(deviceB, m1.data(), m1.getWidth() * m1.getHeight() * sizeof(T), cudaMemcpyHostToDevice))

    gettimeofday(&t2, nullptr);

    double time = (1000000.0*((double )t2.tv_sec-(double)t1.tv_sec) + (double )t2.tv_usec-(double )t1.tv_usec)/1000.0;

    printf("Time to copy from RAM to V-RAM:  %3.1f ms \n", time);

    gettimeofday(&t1, nullptr);
    MatMul<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, this->width, this->height, m1.getWidth(), m1.getHeight(), CWidth, CHeight);
    //addMatrix<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, 3, 3);
    gpuErrorChek(cudaPeekAtLastError())
    gpuErrorChek(cudaDeviceSynchronize())
    gettimeofday(&t2, nullptr);

    time = (1000000.0*((double )t2.tv_sec-(double )t1.tv_sec) + (double )t2.tv_usec-(double )t1.tv_usec)/1000.0;

    printf("Time to calculate on gpu:  %3.1f ms \n", time);

    cudaMemcpy(result.data(), deviceC, CWidth * CHeight * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return result;
}

template<typename T>
T *Matrix<T>::data() {
    return this->matrix;
}

template<typename T>
void Matrix<T>::print() const {
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            cout<<matrix[i*width+j]<<" ";
        }
        cout<<endl;
    }
}

template<typename T>
Matrix<T> Matrix<T>::operator>>(Matrix<T> m1) {
    return m1;
}

template<typename T>
void Matrix<T>::randomize() {
    for(int i = 0; i < this->height; i++){
        for(int j = 0; j < this->width; j++){
            this->data()[i*width+j] = rand()%10;
        }
    }


}

void matrix_mul_cpu_impl(int M, int N, int K, const float * A, const float * B, float * C)
{
    for (int i = 0; i < M; ++i)
    {
        float * c = C + i * N;
        for (int j = 0; j < N; ++j)
            c[j] = 0;
        for (int k = 0; k < K; ++k)
        {
            const float * b = B + k * N;
            float a = A[i*K + k];
            for (int j = 0; j < N; ++j)
                c[j] += a * b[j];
        }
    }
}

template<typename T>
__attribute__((unused))
Matrix<T> Matrix<T>::multiplyCpu(Matrix<T> m1) {
    struct timeval t1{}, t2{};

    gettimeofday(&t1, nullptr);

    Matrix m(this->width, m1.getHeight());
    matrix_mul_cpu_impl(this->width, m1.getHeight(), this->height, this->data(), m1.data(), m.data());
    gettimeofday(&t2, nullptr);

    double time = (1000000.0*((double )t2.tv_sec-(double )t1.tv_sec) + (double )t2.tv_usec-(double )t1.tv_usec)/1000.0;

    printf("Time to calculate on cpu:  %3.1f ms \n", time);

    return m;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(Matrix<T> m1) {
    int CWidth = this->width;
    int CHeight = this->height;
    Matrix result = Matrix(CWidth, CHeight);

    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid;

    dimGrid.x = (CWidth + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (CHeight + dimBlock.y - 1) / dimBlock.y;

    float *deviceA, *deviceB, *deviceC;

    struct timeval t1{}, t2{};

    gettimeofday(&t1, nullptr);


    gpuErrorChek(cudaMalloc((void **) &deviceA, this->width * this->height * sizeof(float)))
    gpuErrorChek(cudaMalloc((void **) &deviceB, m1.getWidth() * m1.getHeight() * sizeof(float)))
    gpuErrorChek(cudaMalloc((void **) &deviceC, CWidth * CHeight * sizeof(float)))

    gpuErrorChek(cudaMemcpy(deviceA, this->data(), this->width * this->height * sizeof(float), cudaMemcpyHostToDevice))
    gpuErrorChek(cudaMemcpy(deviceB, m1.data(), m1.getWidth() * m1.getHeight() * sizeof(float), cudaMemcpyHostToDevice))

    gettimeofday(&t2, nullptr);

    double time = (1000000.0*((double )t2.tv_sec-(double )t1.tv_sec) + (double)t2.tv_usec-(double)t1.tv_usec)/1000.0;

    printf("Time to copy from RAM to V-RAM:  %3.1f ms \n", time);

    gettimeofday(&t1, nullptr);
    addMatrix<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, this->width);
    gpuErrorChek(cudaPeekAtLastError())
    gpuErrorChek(cudaDeviceSynchronize())
    gettimeofday(&t2, nullptr);

    time = (1000000.0*((double )t2.tv_sec-(double )t1.tv_sec) + (double )t2.tv_usec-(double )t1.tv_usec)/1000.0;

    printf("Time to calculate on gpu:  %3.1f ms \n", time);

    cudaMemcpy(result.data(), deviceC, CWidth * CHeight * sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}

template<typename T>
int Matrix<T>::operator()() {
    Matrix<T> m = this->to_triangle();
    double C = 1;
    for(int i = 0; i< this->width; i++){
        C*=m.data()[i * this->width + i];
    }
    return (int)round(C);
}

template<typename T>
Matrix<T> Matrix<T>::solve(Matrix b) {
    T *dev_a, *dev_b;

    cudaMalloc((void **) &dev_a, this->width * this->height * sizeof(T));
    cudaMalloc((void **) &dev_b, b.getWidth() * b.getHeight() * sizeof(T));

    cudaMemcpy(dev_a, this->matrix, this->width * this->height  * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), b.getWidth() * b.getHeight()  * sizeof(T), cudaMemcpyHostToDevice);

    dim3 Grid(ceil(this->width/(TILE_DIM*1.)), ceil(this->height/(TILE_DIM*1.)));
    dim3 Block(TILE_DIM, TILE_DIM);

    cudaEvent_t start, stop;
    float gpuTime = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, nullptr);

    for (int i = 0; i < this->height - 1; i++) {
        kernel_SLAE_To_Triangle<<<Grid, Block>>>(dev_a, dev_b, this->height, i, i);
        gpuErrorChek(cudaPeekAtLastError())
        gpuErrorChek(cudaDeviceSynchronize())
    }
    //for (int i = this->height - 1; i >= 0; i--) kernel_up<<<Grid,Block>>>(dev_a, dev_b, this->height, i, i);

    Matrix<T> resultMatrix(this->width, this->height);
    Matrix<T> resultVector(b.getWidth(), b.getHeight());

    cudaMemcpy(resultMatrix.data(), dev_a,this->width * this->height*sizeof(T),cudaMemcpyDeviceToHost);
    cudaMemcpy(resultVector.data(), dev_a,b.getWidth() * b.getHeight()*sizeof(T),cudaMemcpyDeviceToHost);

    resultMatrix.print();
    resultVector.print();

    //for (int i = 0; i < n - 1; i++) h_C[i]=h_B[i]/h_A[i*n+i];

    cudaEventRecord(stop,nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    //this->print();

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaDeviceReset();

    return Matrix<T>(0, 0);
}



