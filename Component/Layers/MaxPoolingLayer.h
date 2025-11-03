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
        // Constructor for deserialization
       MaxPoolingLayer() : stride(1), size(2) {}

        void inference(void *ptr_lastLayerInput) override {
           Matrix<Tracebackable<float>>* inputMatrix = static_cast<Matrix<Tracebackable<float>>*>(ptr_lastLayerInput);
           // std::cout << "input matrix \n" << *inputMatrix;
            auto outputMatrix = inputMatrix->maxPooling(size,stride);
          //  std::cout << "output matrix \n" << *outputMatrix;
            setOutActivation(outputMatrix);
            if (nextLayer != nullptr)
                nextLayer->inference(outputMatrix);
        }
        // reShape the layer
        void backward(void *ptr_nextLayerInput,float learningRate) override {
            Matrix<float>* ptr_nextLayerWeight = static_cast<Matrix<float>*>(ptr_nextLayerInput);
            dim3 thisActivationDim = getOutActivation()->getDim();
            ptr_nextLayerWeight->reShape(thisActivationDim.x,thisActivationDim.y,thisActivationDim.z);
            if (lastLayer != nullptr) {
                ConvBackwardData backward_data = ConvBackwardData { getOutActivation(),ptr_nextLayerWeight };
                lastLayer->backward(&backward_data,learningRate);
            }
        }
    };
}
#endif //RECNN_MAXPOOLINGLAYER_H