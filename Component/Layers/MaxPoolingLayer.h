//
// Created by datdau on 10/26/25.
//

#ifndef RECNN_MAXPOOLINGLAYER_H
#define RECNN_MAXPOOLINGLAYER_H
#include "LayerBase.h"
#include "../Matrix.h"

namespace TommyDat {
    class MaxPoolingLayer : public Layer {
    public:
        int stride;
        int size;
        int inputHeight = 0;
        int inputWidth  = 0;
        MaxPoolingLayer(int stride,int size) {
            this->stride = stride;
            this->size = size;
        }

        int getStride() const { return
            stride;

        }
        void setStride(int s) {
            stride = s;

        }
        int getSize() const { return
            size;

        }
        void setSize(int s) {
            size = s;
        }

        //Input shape for easier image loads
        int getInputHeight() const {
            return inputHeight;
        }

        int getInputWidth() const {
            return inputWidth;
        }


        int getChannels() const {
            if (lastLayer) {
                if (auto conv = dynamic_cast<ConvolutionLayer*>(lastLayer)) {
                    return conv->getOutChannel();
                } else if (auto pool = dynamic_cast<MaxPoolingLayer*>(lastLayer)) {
                    return pool->getChannels();
                }
            }
            return 1; // fallback
        }

        void setInputShape(int h, int w) {
            inputHeight = h;
            inputWidth = w;
        }

        int getOutputSizeFlattened() const override {
            if (inputHeight == 0 || inputWidth == 0)
                throw std::runtime_error("MaxPoolingLayer: input shape not set");

            int H_out = inputHeight / stride; // integer division
            int W_out = inputWidth / stride;

            // Assuming channel count comes from lastLayer
            int channels = 1;
            if (lastLayer) {
                if (auto conv = dynamic_cast<ConvolutionLayer*>(lastLayer)) {
                    channels = conv->getOutChannel();
                } else if (auto pool = dynamic_cast<MaxPoolingLayer*>(lastLayer)) {
                    channels = pool->getChannels(); // add getter in MaxPoolingLayer
                }
            }

            return channels * H_out * W_out;
        }

        // Constructor for deserialization
       MaxPoolingLayer() : stride(1), size(2) {}

        void inference(void *ptr_lastLayerInput) override {
           Matrix<Tracebackable<float>>* inputMatrix = static_cast<Matrix<Tracebackable<float>>*>(ptr_lastLayerInput);
         //   std::cout << "INPUT MAX POOL \n" << *inputMatrix;
            // setInActivation(inputMatrix);
            auto maxPoolInp = inputMatrix->maxPooling(size,stride);
            Matrix<Tracebackable<float>>* outputMatrix = new Matrix(maxPoolInp);
           //std::cout << "OUTPUT MAX POOL \n" << *outputMatrix;
            setOutActivation(outputMatrix);
            if (nextLayer != nullptr)
                nextLayer->inference(outputMatrix);
        }
        // reShape the layer
        void backward(void *ptr_nextLayerInput,float learningRate) override {
            Matrix<float>* ptr_nextLayerWeight = static_cast<Matrix<float>*>(ptr_nextLayerInput);
            void* ptrvoid_OutAct = getOutActivation();
            auto ptr_outAct = static_cast<Matrix<Tracebackable<float>>*>(ptrvoid_OutAct);
            dim3 thisActivationDim = ptr_nextLayerWeight->getDim();
            ptr_nextLayerWeight->reShape(thisActivationDim.x,thisActivationDim.y,thisActivationDim.z);


            if (lastLayer != nullptr) {
                ConvBackwardData backward_data = ConvBackwardData { ptr_outAct,ptr_nextLayerWeight };
                lastLayer->backward(&backward_data,learningRate);
            }
        }
    };
}
#endif //RECNN_MAXPOOLINGLAYER_H