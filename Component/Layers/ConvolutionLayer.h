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

        int inputHeight = 0;
        int inputWidth  = 0;
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

        Matrix<float>* getKernelMatrix() const {
            return kernelList;
        }

        void setKernelMatrix(const Matrix<float>& mat) {
            if (kernelList != nullptr) {
                delete kernelList;
            }
            kernelList = new Matrix<float>(mat); // allocate fresh copy
        }

        // GETTER SETTER for serializing
        int getInChannel() const {
            return inChannel;
        }
        int getOutChannel() const {
            return outChannel;
        }
        int getKernelSize() const {
            return kernelSize;
        }

        //Input shape for easier image loads
        int getInputHeight() const {
            return inputHeight;
        }

        int getInputWidth() const { return
            inputWidth; }

        void setInputShape(int h, int w) {
            inputHeight = h;
            inputWidth = w;
        }

        int getOutputSizeFlattened() const override {
            if (inputHeight == 0 || inputWidth == 0)
                throw std::runtime_error("ConvolutionLayer: input shape not set");

            int H_out = (inputHeight - kernelSize) / stride + 1;
            int W_out = (inputWidth - kernelSize) / stride + 1;
            return outChannel * H_out * W_out;
        }

        void inference(void* ptr_lastLayerInput) override {
            Matrix<Tracebackable<float>>* inputMatrix = static_cast<Matrix<Tracebackable<float>>*>(ptr_lastLayerInput);
            Matrix<Tracebackable<float>>* output = inputMatrix->convolution(*kernelList,stride);
           //   std::cout << "INPUT CONV MATRIX \n" << *inputMatrix;
             // std::cout << "output matrix \n" << *output;
            output->ReLU();
            setInActivation(inputMatrix);
            if (nextLayer != nullptr)
                nextLayer->inference(output);
        }
        void printWeight() {
            std::cout << "Kernel list \n" << *kernelList;
        }
        void backward(void* ptr_nextLayerInput,float leaningRate) override {
            ConvBackwardData* ptr_nextLayer = static_cast<ConvBackwardData*>(ptr_nextLayerInput);
            void* ptrvoid_inAct = getInActivation();
            auto ptr_inAct = static_cast<Matrix<Tracebackable<float>>*>(ptrvoid_inAct);
            float* ptr_newWeightRaw = CallGPUConvBackward(ptr_nextLayer->ptr_nextLayerPixelAffect->flatten(),ptr_inAct->flatten(),ptr_nextLayer->gradient->flatten(),kernelList->flatten(),
                ptr_inAct->getDim() ,ptr_nextLayer->gradient->getDim(),
                inChannel,kernelSize,leaningRate);
            dim3 inputMatrixDim = ptr_inAct->getDim();
            Matrix newWeightMat = Matrix(ptr_newWeightRaw,inputMatrixDim);
           // std::cout << "CONVOUTION BACKWARD" << *ptr_nextLayer->ptr_nextLayerPixelAffect << "weight:\n" << *ptr_nextLayer->gradient << "newWeight:\n" << newWeightMat << "new Ker\n" << *kernelList;
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
