cmake_minimum_required(VERSION 3.20)
project(cudaMatrixCalculator CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cudaMatrixCalculator main.cu Matrix.cuh)

set_target_properties(cudaMatrixCalculator PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
