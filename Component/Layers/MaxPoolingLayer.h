//
// Created by datdau on 10/26/25.
//

#ifndef RECNN_MAXPOOLINGLAYER_H
#define RECNN_MAXPOOLINGLAYER_H
#include "LayerBase.h"
#include "../Kernel3D.h"
namespace TommyDat {
    class MaxPoolingLayer : Layer {
    public:
        int stride;
        int size;
        MaxPoolingLayer(int stride,int size) {
            this->stride = stride;
            this->size = size;
        }
        void inference(void *ptr_lastLayerInput) override {
            Kernel3D<float>* inputMatrix = static_cast<Kernel3D<float>*>(ptr_lastLayerInput);
            dim3 inputMatrixDim = inputMatrix->getDim();
            for (int i = 0;i < inputMatrixDim.x;i++) {
                Matrix output = new Matrix(inputMatrix->data[i]->maxPooling(size,stride));
                ptr_listOutputMatrix[i] = output;
            }

            Kernel3D outputMatrix(ptr_listOutputMatrix,outChannel,inputMatrixDim.x,inputMatrixDim.y);

            if (nextLayer != nullptr)
                nextLayer->inference(&outputMatrix);
        }
        void backward(void *nextLayerInput) override {

        }
    };
}
#endif //RECNN_MAXPOOLINGLAYER_H