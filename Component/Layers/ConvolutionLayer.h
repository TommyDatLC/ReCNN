//
// Created by datdau on 10/26/25.
//
#ifndef RECNN_CONVOLUTIONLAYE_H
#define RECNN_CONVOLUTIONLAYE_H
#include "LayerBase.h"
#include "../Kernel3D.h"

namespace TommyDat {
    class ConvolutionLayer : public Layer {
        int inChannel;
        int outChannel;
        int kernelSize;
        int stride;
        Kernel3D<float>* kernelList;
        Matrix<float>** ptr_listOutputMatrix ;
    // Convolution sum to nextLayer
    public:
        ConvolutionLayer(int inChannel,int outChannel,int kernelSize,int stride) {
            this->inChannel = inChannel;
            this->outChannel = outChannel;
            this->kernelSize = kernelSize;
            this->stride = stride;
            ptr_listOutputMatrix = new Matrix<float>*[outChannel];
            if (outChannel % inChannel != 0) {
                throw std::runtime_error("outChannel must be a mutiple with inChannel");
            }
            kernelList = new Kernel3D<float>(outChannel,kernelSize,kernelSize);
        }

        void inference(void* ptr_lastLayerInput) override {
            Kernel3D<float>* inputMatrix = static_cast<Kernel3D<float>*>(ptr_lastLayerInput);

            for (int i = 0;i < outChannel;i++) {
                int idInputMatrix = i % inChannel;
                std::cout << (inputMatrix->data[idInputMatrix]->flatten() == nullptr);
                Matrix output = inputMatrix->data[idInputMatrix]->convolution(*kernelList->data[i]);
                ptr_listOutputMatrix[i] = &output;
            }

            dim3 outMatrixDim = ptr_listOutputMatrix[0]->getDim();
            Kernel3D outputMatrix(ptr_listOutputMatrix,outChannel,outMatrixDim.x,outMatrixDim.y);

            if (nextLayer != nullptr)
                nextLayer->inference(&outputMatrix);
        }
        void backward(void* nextLayerInput) override {

        }
        ~ConvolutionLayer() {
            delete kernelList;
        }
    };

}
#endif //RECNN_CONVOLUTIONLAYE_H
