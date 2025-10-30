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
        float getError(void *predictResult) override {
            Matrix<Tracebackable<float>>* predictResultMatrix = static_cast<Matrix<Tracebackable<float>>*>(predictResult);
            dim3 predictResultMatrixDim = predictResultMatrix->getDim();
            Matrix labelMatrix = Matrix<Tracebackable<float>>(predictResultMatrixDim.x,predictResultMatrixDim.y,predictResultMatrixDim.z,0.f);
            labelMatrix.setFlatten(lable,1.f);

            predictResultMatrix->log();
            auto ptr_afterMul =  Matrix<Tracebackable<float>>::mulUnofficial(*predictResultMatrix,labelMatrix);
            std::cout << *ptr_afterMul;
            Tracebackable<float> res = CallGPUSum(ptr_afterMul->flatten(),predictResultMatrixDim.x * predictResultMatrixDim.y * predictResultMatrixDim.z);
            delete ptr_afterMul;
            return res.get();
        }
            // return -logf( predictResultMatrix->getFlatten(lable).get());
    };
}
#endif //RECNN_NEURALINPUT_H
