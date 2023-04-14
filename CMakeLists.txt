cmake_minimum_required(VERSION 3.20)
project(cudaMatrixCalculator CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cudaMatrixCalculator main.cu)

include_directories(kernels)

set_target_properties(cudaMatrixCalculator PROPERTIES
        CUDA_ARCHITECTURES "86"
        CUDA_SEPARABLE_COMPILATION ON)
