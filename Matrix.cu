//
// Created by artyom on 9/8/21.
//

#include "Matrix.cuh"



__global__ void addMatrix(const float* A, const float* B, float* C, int Cols){
    unsigned int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    unsigned int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    C[Row*Cols+Col] = A[Row*Cols+Col] + B[Row*Cols+Col];
}

template<typename T>
__global__ void
MatMul(const T* A,const T* B, T* C, int ARows, int ACols, int BRows,
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

template <typename T>
std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> Matrix<T>::PLU_factorisation() {
    T *dev_p, *dev_l, *dev_u, *dev_max_val;
    unsigned int *dev_max_idx;

    Matrix<T> P(this->width, this->height);
    Matrix<T> L(this->width, this->height);
    Matrix<T> U(this->width, this->height);
    P.eye();
    auto t1 = high_resolution_clock::now();
    cudaMalloc((void **) &dev_p, this->width * this->height * sizeof(T ));
    cudaMalloc((void **) &dev_l, this->width * this->height * sizeof(T ));
    cudaMalloc((void **) &dev_u, this->width * this->height * sizeof(T ));

    cudaMalloc((void **) &dev_max_val, max(this->height/(TILE_DIM), 1) * sizeof(T ));
    cudaMalloc((void **) &dev_max_idx, max(this->height/(TILE_DIM), 1) * sizeof(int));

    cudaMemcpy(dev_u, this->matrix, this->width * this->height  * sizeof(T), cudaMemcpyHostToDevice);



    cudaMemcpy(dev_p, P.data(), this->width * this->height  * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_l, P.data(), this->width * this->height  * sizeof(T), cudaMemcpyHostToDevice);


    auto t2 = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(t2 - t1);
    printf("Time to copy from RAM to V-RAM: %d us %1.5f s \n", time.count(), time.count()*1e-6);
    dim3 Grid(ceil(this->width/(TILE_DIM*1.)), ceil(this->height/(TILE_DIM*1.)));
    dim3 Block(TILE_DIM, TILE_DIM);

    t1 = high_resolution_clock::now();
    for (int i = 0; i < this->width-1; i++) {
        kernel_max_mod_cols<<<Grid, Block>>>(dev_u, dev_max_val, dev_max_idx, i, this->width, this->height);
        gpuErrorChek(cudaPeekAtLastError());
        gpuErrorChek(cudaDeviceSynchronize());

        unsigned int *maxIdx = (unsigned int*)malloc(max(this->height/(TILE_DIM), 1) * sizeof(unsigned int));
        cudaMemcpy(maxIdx, dev_max_idx,max(this->height/(TILE_DIM), 1) * sizeof(unsigned int),cudaMemcpyDeviceToHost);

        kernel_swap_rows_and_simplify<<<Grid, Block>>>(dev_u, dev_p, dev_l, i, maxIdx[0], this->width, this->height);
        gpuErrorChek(cudaPeekAtLastError());
        gpuErrorChek(cudaDeviceSynchronize());

    }
    t2 = high_resolution_clock::now();
    time = duration_cast<microseconds>(t2 - t1);
    printf("Time  to calculate on gpu: %d us %1.5f s \n", time.count(), time.count()*1e-6);
    t1 = high_resolution_clock::now();
    cudaMemcpy(U.data(), dev_u, this->width * this->height  * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(L.data(), dev_l, this->width * this->height  * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(P.data(), dev_p, this->width * this->height  * sizeof(T), cudaMemcpyDeviceToHost);
    t2 = high_resolution_clock::now();
    time = duration_cast<microseconds>(t2 - t1);
    printf("Time to copy from V-RAM to RAM: %d us %1.5f s \n", time.count(), time.count()*1e-6);
    cudaFree(dev_u);
    cudaDeviceReset();

    return std::tuple<Matrix, Matrix, Matrix>(P, L, U);
}

template<typename T>
template< typename... Args, typename>
Matrix<T>::Matrix(int width, int height, Args... args) {
    this->width = width;
    this->height = height;
    this->matrix=(T *) malloc(width * height * sizeof(T));

    T res[sizeof...(Args)] = {args...};
    std::memcpy(this->matrix, res, width * height * sizeof(T));
 //   matrix[0] = ;
//    for(int i = 0; i<height;i++){
//        for(int j = i==0?1:0; j<width;j++){
//            matrix[i*width+j] = va_arg(args, T);
//        }
//    }
//    va_end(args);
}

template<typename T>
Matrix<T>::Matrix(int width, int height) {
    this->width = width;
    this->height = height;
    this->matrix=(T *) calloc(width * height, sizeof(T));
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

    auto t1 = high_resolution_clock::now();


    gpuErrorChek(cudaMalloc((void **) &deviceA, this->width * this->height * sizeof(T)))
    gpuErrorChek(cudaMalloc((void **) &deviceB, m1.getWidth() * m1.getHeight() * sizeof(T)))
    gpuErrorChek(cudaMalloc((void **) &deviceC, CWidth * CHeight * sizeof(T)))

    gpuErrorChek(cudaMemcpy(deviceA, this->data(), this->width * this->height * sizeof(T), cudaMemcpyHostToDevice))
    gpuErrorChek(cudaMemcpy(deviceB, m1.data(), m1.getWidth() * m1.getHeight() * sizeof(T), cudaMemcpyHostToDevice))

    auto t2 = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(t2 - t1);

    printf("Time to copy from RAM to V-RAM:  %d us %1.5f s \n", time.count(), time.count()*1e-6);

    t1 = high_resolution_clock::now();
    MatMul<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, this->width, this->height, m1.getWidth(), m1.getHeight(), CWidth, CHeight);
    //addMatrix<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, 3, 3);
    gpuErrorChek(cudaPeekAtLastError())
    gpuErrorChek(cudaDeviceSynchronize())
    t2 = high_resolution_clock::now();

    time = duration_cast<microseconds>(t2 - t1);

    printf("Time to calculate on gpu:  %d us %1.5f s \n", time.count(), time.count()*1e-6);

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
            cout << std::fixed<< std::setprecision(6)<<matrix[i*width+j]<<"\t";
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
Matrix<T> Matrix<T>::multiplyCpu(Matrix<T> m1) {
    auto t1 = high_resolution_clock::now();

    Matrix m(this->width, m1.getHeight());
    matrix_mul_cpu_impl(this->width, m1.getHeight(), this->height, this->data(), m1.data(), m.data());

    auto t2 = high_resolution_clock::now();

    auto time = duration_cast<microseconds>(t2 - t1);

    printf("Time to calculate on cpu:  %d us %1.5f s \n", time.count(), time.count()*1e-6);

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

    auto t1 = high_resolution_clock::now();


    gpuErrorChek(cudaMalloc((void **) &deviceA, this->width * this->height * sizeof(float)))
    gpuErrorChek(cudaMalloc((void **) &deviceB, m1.getWidth() * m1.getHeight() * sizeof(float)))
    gpuErrorChek(cudaMalloc((void **) &deviceC, CWidth * CHeight * sizeof(float)))

    gpuErrorChek(cudaMemcpy(deviceA, this->data(), this->width * this->height * sizeof(float), cudaMemcpyHostToDevice))
    gpuErrorChek(cudaMemcpy(deviceB, m1.data(), m1.getWidth() * m1.getHeight() * sizeof(float), cudaMemcpyHostToDevice))

    auto t2 = high_resolution_clock::now();

    auto time = duration_cast<microseconds>(t2 - t1);

    printf("Time to copy from RAM to V-RAM:  %d us %1.5f s \n", time.count(), time.count()*1e-6);

    t1 = high_resolution_clock::now();
    addMatrix<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, this->width);
    gpuErrorChek(cudaPeekAtLastError())
    gpuErrorChek(cudaDeviceSynchronize())
    t2 = high_resolution_clock::now();

    time = duration_cast<microseconds>(t2 - t1);

    printf("Time to calculate on gpu: %d us %1.5f s \n", time.count(), time.count()*1e-6);

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
std::tuple<Matrix<T>, Matrix<T>> Matrix<T>::solve(Matrix b) {
    T *dev_a, *dev_b;

    cudaMalloc((void **) &dev_a, this->width * this->height * sizeof(T));
    cudaMalloc((void **) &dev_b, b.getWidth() * b.getHeight() * sizeof(T));

    cudaMemcpy(dev_a, this->matrix, this->width * this->height  * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), b.getWidth() * b.getHeight()  * sizeof(T), cudaMemcpyHostToDevice);

    dim3 Grid(max(this->width/(TILE_DIM), 1), max(this->height/(TILE_DIM), 1));
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
    kernel_up<<<Grid,Block>>>(dev_a, dev_b, this->height);


    Matrix<T> resultVector(b.getWidth(), b.getHeight());
    Matrix<T> resultMatrix(this->width, this->height);

    cudaMemcpy(resultMatrix.data(), dev_a,this->width * this->height*sizeof(T),cudaMemcpyDeviceToHost);
    cudaMemcpy(resultVector.data(), dev_b,b.getWidth() * b.getHeight()*sizeof(T),cudaMemcpyDeviceToHost);

//    resultMatrix.print();
//    resultVector.print();

    //for (int i = 0; i < n - 1; i++) h_C[i]=h_B[i]/h_A[i*n+i];

    cudaEventRecord(stop,nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    //this->print();

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaDeviceReset();

    return std::tuple<Matrix<T>, Matrix<T>>(resultMatrix, resultVector);
}

template <typename T> Matrix<T> Matrix<T>::forward_substitution(Matrix<T> b) {
    int m = b.getWidth();
    Matrix<T> x(m, 1);
    for(int i = 0; i < m; i++){
        x.data()[i] = 1;
        if(abs(this->data()[i* this->width+i])<1E-7){
            x.data()[i] = 0;
        }
        T val = b.data()[i];
        for(int j = 0; j < i; j++){
            val -= this->data()[i* this->width+j] * x.data()[j];
        }
        val /= this->data()[i* this->width+i];

        x.data()[i] = val;
    }
    return x;
}

template <typename T> Matrix<T> Matrix<T>::backward_substitution(Matrix<T> b) {
    int m = b.getWidth();
    Matrix<T> x(m, 1);
    for(int i = m; i > -1; i--){
        x.data()[i] = 1;
        if(abs(this->data()[i* this->width+i])<1E-7){
            x.data()[i] = 0;
        }
        T val = b.data()[i];
        for(int j = i+1; j < m; j++){
            val -= this->data()[i* this->width+j] * x.data()[j];
        }
        val /= this->data()[i* this->width+i];

        x.data()[i] = val;
    }
    return x;
}

template <typename T> void Matrix<T>::eye() {
    for(int i = 0; i < min(this->width, this->height); i++){
        this->matrix[i* this->width+i] = 1;
    }
}



