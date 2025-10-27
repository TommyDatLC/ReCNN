//
// Created by datdau on 10/26/25.
//

#ifndef RECNN_MAXPOOLINGLAYER_H
#define RECNN_MAXPOOLINGLAYER_H
#include "LayerBase.h"
#include "../Kernel3D.h"
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

            Kernel3D<float>* inputMatrix = static_cast<Kernel3D<float>*>(ptr_lastLayerInput);
            dim3 inputMatrixDim = inputMatrix->getDim();
            ptr_listOutputMatrix = new Matrix<float> *[inputMatrixDim.x];
            for (int i = 0;i < inputMatrixDim.x;i++) {
                Matrix poolingResult = inputMatrix->data[i]->maxPooling(size,stride);
                Matrix<float>* output = new Matrix(poolingResult);
                ptr_listOutputMatrix[i] = output;
            }

            Kernel3D outputMatrix(ptr_listOutputMatrix,inputMatrixDim.x ,inputMatrixDim.y,inputMatrixDim.z);

            if (nextLayer != nullptr)
                nextLayer->inference(&outputMatrix);

        }
        void backward(void *nextLayerInput) override {

        }
    private:
        Matrix<float>** ptr_listOutputMatrix;
    };
}
#endif //RECNN_MAXPOOLINGLAYER_H