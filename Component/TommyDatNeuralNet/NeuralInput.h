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
        Matrix<float>* getErrorMatrix(void* lastLayerActivation) override {
            Matrix<Tracebackable<float>>* predictResultMatrix = static_cast<Matrix<Tracebackable<float>>*>(lastLayerActivation);
            dim3 predictResultMatrixDim = predictResultMatrix->getDim();
            Matrix labelMatrix = Matrix(predictResultMatrixDim,0.f);
            labelMatrix.setFlatten(lable,-1.f);
            Matrix<float>* result =  labelMatrix + *predictResultMatrix;
            return result;
        }
        float getError(void *predictResult) override {
            Matrix<Tracebackable<float>>* predictResultMatrix = static_cast<Matrix<Tracebackable<float>>*>(predictResult);

            Matrix<float> predictResultValueMatrix = toValueMatrix<float>(*predictResultMatrix);
            dim3 predictResultMatrixDim = predictResultValueMatrix.getDim();
            Matrix labelMatrix = Matrix(predictResultMatrixDim,0.f);
            labelMatrix.setFlatten(lable,1.f);
            predictResultMatrix->log();
            auto ptr_afterMul =  mulUnofficial(labelMatrix,predictResultValueMatrix);
           // std::cout << *ptr_afterMul;
            // std::cout << *ptr_afterMul;
            float res = CallGPUSum(ptr_afterMul->flatten(),predictResultMatrixDim.x * predictResultMatrixDim.y * predictResultMatrixDim.z);
            delete ptr_afterMul;
            return res;
        }
            // return -logf( predictResultMatrix->getFlatten(lable).get());
    };
}
#endif RECNN_NEURALINPUT_H
