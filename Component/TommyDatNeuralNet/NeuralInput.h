//
// Created by datdau on 10/26/25.
//

#ifndef RECNN_NEURALINPUT_H
#define RECNN_NEURALINPUT_H
#include "NeuralInputBase.h"

namespace TommyDat {
    class NeuralInput : public NeuralInputBase<Matrix<Tracebackable<float>>> {
    public:
        NeuralInput(std::string path) : NeuralInputBase() {
            data = new Matrix<Tracebackable<float>>(path);
            data->normalize();

            // std::cout << "input:\n" << *data;
        }
        NeuralInput() {

        }
        Matrix<float> getGradientMatrix(void* lastLayerActivation) override {
            Matrix<float>* predictResultMatrix = static_cast<Matrix<float>*>(lastLayerActivation);
            dim3 predictResultMatrixDim = predictResultMatrix->getDim();
            Matrix labelMatrix = Matrix(predictResultMatrixDim,0.f);
            labelMatrix.setFlatten(lable,-1.f);
            Matrix<float> result =  labelMatrix + *predictResultMatrix;

            return result;
        }
        float getError(void *predictResult) override {
           Matrix<float>* predictResultMatrix = static_cast<Matrix<float>*>(predictResult);


            float res = -logf( predictResultMatrix->getFlatten(lable));

            return res;
        }
    };
}
#endif RECNN_NEURALINPUT_H
