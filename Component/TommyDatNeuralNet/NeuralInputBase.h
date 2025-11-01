//
// Created by datdau on 10/26/25.
//

#ifndef RECNN_NEURALINPUTBASE_H
#define RECNN_NEURALINPUTBASE_H
#include "../Matrix.h"

namespace TommyDat {
    template <typename Tdata>
    class NeuralInputBase {
    public:
        Tdata* data;
        int lable = 5;
        virtual float getError(void* predictResult) = 0;
        virtual Matrix<float>* getGradientMatrix(void* lastLayerActivation) = 0;
    };
}

#endif //RECNN_NEURALINPUTBASE_H