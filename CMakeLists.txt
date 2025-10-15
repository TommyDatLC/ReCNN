cmake_minimum_required(VERSION 4.0)
project(ReCNN CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(ReCNN
        NNLayer.cuh
        NN.cuh
        Utils/Utility.cuh
        main.cu
        Utils/GPUmax.h
        Utils/GPUPrefixSum.h
)

set_target_properties(ReCNN PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
