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
            Matrix<float>* inputMatrix = static_cast<Matrix<float>*>(ptr_lastLayerInput);
            auto outputMatrix = inputMatrix->maxPooling(stride,size);
            setNewActivation(outputMatrix);
            if (nextLayer != nullptr)
                nextLayer->inference(outputMatrix);
        }
        void backward(void *nextLayerInput) override {

        }
    };
}
#endif //RECNN_MAXPOOLINGLAYER_H