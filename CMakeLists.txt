cmake_minimum_required(VERSION 3.16)
project(cudaMatrixCalculator CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE on)

add_compile_options(--expt-relaxed-constexpr)

add_executable(cudaMatrixCalculator main.cu kernels/PLUfactorisation.cuh)

include_directories(kernels)

set_target_properties(cudaMatrixCalculator PROPERTIES
        CUDA_ARCHITECTURES "86"
        CUDA_SEPARABLE_COMPILATION ON)
