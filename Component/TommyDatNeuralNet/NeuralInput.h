//
// Created by datdau on 10/26/25.
//

#ifndef RECNN_NEURALINPUT_H
#define RECNN_NEURALINPUT_H
#include "NeuralInputBase.h"
#include "../Kernel3D.h"

namespace TommyDat {
    class NeuralInput : NeuralInputBase<Kernel3D<int>> {
        void ReadData() {
            data = Kernel3D<int>("");
        }
    };
}
#endif //RECNN_NEURALINPUT_H
