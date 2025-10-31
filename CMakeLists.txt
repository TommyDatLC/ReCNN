cmake_minimum_required(VERSION 4.0)
project(ReCNN CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(ReCNN
        Utils/Utility.cuh
        main.cu
        Utils/GPUmax.h
        Utils/GPUPrefixSum.h
        Utils/GPUMatrixOp.h
        Component/Matrix.h
        Utils/GPUSoftMax.h
        Utils/Sum.h
        Component/EnumActivationType.h
        Component/TommyDatNeuralNet/NeuralNetwork.h
        Component/Layers/LayerBase.h
        Component/Layers/ConvolutionLayer.h
        Component/TommyDatNeuralNet/NeuralInputBase.h
        Component/TommyDatNeuralNet/NeuralInput.h
        Component/Layers/MaxPoolingLayer.h
        Component/NNlayer.h
        Component/Tracebackable.h
        Component/ConvolutionLayerBackward.h
        Component/TommyDatNeuralNet/NeuralInput.h
        Utils/stbImage.h

)


set_target_properties(ReCNN PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

file(COPY testdata1.png DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
