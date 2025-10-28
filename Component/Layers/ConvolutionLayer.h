//
// Created by datdau on 10/26/25.
//
#ifndef RECNN_CONVOLUTIONLAYE_H
#define RECNN_CONVOLUTIONLAYE_H
#include "LayerBase.h"


namespace TommyDat {
    class ConvolutionLayer : public Layer {
        int inChannel;
        int outChannel;
        int kernelSize;
        int stride;
        Matrix<float>* kernelList;

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
            kernelList = new Matrix<float>(outChannel,kernelSize,kernelSize);
        }

        void inference(void* ptr_lastLayerInput) override {
            Matrix<float>* inputMatrix = static_cast<Matrix<float>*>(ptr_lastLayerInput);
            Matrix<float>* output = inputMatrix->convolution(*kernelList,stride);
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
