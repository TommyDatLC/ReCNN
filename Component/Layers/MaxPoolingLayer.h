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
        void inference(void *ptr_lastLayerInput) override {
           Matrix<Tracebackable<float>>* inputMatrix = static_cast<Matrix<Tracebackable<float>>*>(ptr_lastLayerInput);
           // std::cout << "input matrix \n" << *inputMatrix;
            auto outputMatrix = inputMatrix->maxPooling(size,stride);
          //  std::cout << "output matrix \n" << *outputMatrix;
            setNewActivation(outputMatrix);
            if (nextLayer != nullptr)
                nextLayer->inference(outputMatrix);
        }
        void backward(void *nextLayerInput) override {

        }
    };
}
#endif //RECNN_MAXPOOLINGLAYER_H