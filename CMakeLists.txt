cmake_minimum_required(VERSION 3.29)
project(untitled4 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(untitled4 main.cu
        main.cu)

set_target_properties(untitled4 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
