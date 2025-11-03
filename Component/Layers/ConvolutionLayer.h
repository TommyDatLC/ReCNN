//
// Created by datdau on 10/26/25.
//
#ifndef RECNN_CONVOLUTIONLAYE_H
#define RECNN_CONVOLUTIONLAYE_H
#include "LayerBase.h"
#include "../ConvolutionLayerBackward.h"
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
          // std::cout << "Kernel list before \n" << *kernelList;
        }
        //Default contructor with zeros values for deserialization
        ConvolutionLayer() : inChannel(0), outChannel(0), kernelSize(0), stride(1) {
            kernelList = nullptr;
        }

        int getStride() const {
            return stride;

        }
        void setStride(int s) {
            stride = s;
        }

        Matrix<float>& getWeightMatrix() const {
            return *kernelList;
        }

        void setWeightMatrix(const Matrix<float>& mat) {
            delete kernelList;
            *kernelList = mat;
        }

        void inference(void* ptr_lastLayerInput) override {
            Matrix<Tracebackable<float>>* inputMatrix = static_cast<Matrix<Tracebackable<float>>*>(ptr_lastLayerInput);
            Matrix<Tracebackable<float>>* output = inputMatrix->convolution(*kernelList,stride);
            output->ReLU();
            // std::cout << "input matrix \n" << *inputMatrix;
            // std::cout << "output matrix \n" << *output;
            setInActivation(inputMatrix);
            if (nextLayer != nullptr)
                nextLayer->inference(output);
        }
        void printWeight() {
            std::cout << "Kernel list \n" << *kernelList;
        }
        void backward(void* ptr_nextLayerInput,float leaningRate) override {
            ConvBackwardData* ptr_nextLayer = static_cast<ConvBackwardData*>(ptr_nextLayerInput);
            float* ptr_newWeightRaw = CallGPUConvBackward(ptr_nextLayer->ptr_nextLayerPixelAffect->flatten(),getInActivation()->flatten(),ptr_nextLayer->gradient->flatten(),kernelList->flatten(),
                getInActivation()->getDim() ,ptr_nextLayer->gradient->getDim(),
                inChannel,kernelSize,leaningRate);
            dim3 inputMatrixDim = getInActivation()->getDim();
            Matrix newWeightMat = Matrix(ptr_newWeightRaw,inputMatrixDim);
            //std::cout << *getInActivation() << "weight:\n" << *ptr_nextLayer->weight << "newWeight:\n" << newWeightMat << "new Ker\n" << *kernelList;
            // 0:2:2
            //
            if (lastLayer != nullptr)
                lastLayer->backward(&newWeightMat,leaningRate);
        }
        ~ConvolutionLayer() {
            delete kernelList;
        }
    };

}
#endif //RECNN_CONVOLUTIONLAYE_H
