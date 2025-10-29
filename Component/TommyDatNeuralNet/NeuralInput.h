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
    };
}
#endif //RECNN_NEURALINPUT_H
