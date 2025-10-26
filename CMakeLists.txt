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
        Utils/GPUMatrixOp.h
        Component/Matrix.h
        Utils/GPUSoftMax.h
        Utils/Sum.h
        Component/IFlattenable.h
        Component/Kernel3D.h
        Component/EnumActivation.h
        Component/TommyDatNeuralNet/NeuralNetwork.h
        Component/Layers/LayerBase.h
        Component/Layers/ConvolutionLayer.h
        Component/TommyDatNeuralNet/NeuralInputBase.h
        Component/TommyDatNeuralNet/NeuralInput.h
        Component/Layers/MaxPoolingLayer.h
)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
set_target_properties(ReCNN PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ReCNN ${OpenCV_LIBS})
