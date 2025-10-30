//
// Created by datdau on 10/26/25.
//
#ifndef RECNN_CONVOLUTIONLAYE_H
#define RECNN_CONVOLUTIONLAYE_H
#include "LayerBase.h"
#include "../EnumActivationType.h"
#include "../Tracebackable.h"


namespace TommyDat {
    // Default activation type is relu
    class ConvolutionLayer : public Layer {
        int inChannel;
        int outChannel;
        int kernelSize;
        int stride;
//
        Matrix<Tracebackable<float>>* kernelList;

    // Convolution sum to nextLayer
    public:
        ConvolutionLayer(int inChannel,int outChannel,int kernelSize,int stride) {
            this->inChannel = inChannel;
            this->outChannel = outChannel;
            this->kernelSize = kernelSize;
            this->stride = stride;
            if (outChannel % inChannel != 0) {
                throw std::runtime_error("outChannel must be a mutiple with inChannel");
            }
            kernelList = new Matrix<Tracebackable<float>>(outChannel,kernelSize,kernelSize);
           // std::cout << "Kernel list \n" << *kernelList;
        }

        void inference(void* ptr_lastLayerInput) override {
            Matrix<Tracebackable<float>>* inputMatrix = static_cast<Matrix<Tracebackable<float>>*>(ptr_lastLayerInput);
            Matrix<Tracebackable<float>>* output = inputMatrix->convolution(*kernelList,stride);
            output->ReLU();
            // std::cout << "input matrix \n" << *inputMatrix;
            // std::cout << "output matrix \n" << *output;
            setNewActivation(output);
            if (nextLayer != nullptr)
                nextLayer->inference(output);
        }
        void backward(void* nextLayerInput) override {

        }
        ~ConvolutionLayer() {
            delete kernelList;
        }
    };

}
#endif //RECNN_CONVOLUTIONLAYE_H
