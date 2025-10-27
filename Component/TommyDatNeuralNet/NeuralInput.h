//
// Created by datdau on 10/26/25.
//

#ifndef RECNN_NEURALINPUT_H
#define RECNN_NEURALINPUT_H
#include "NeuralInputBase.h"
#include "../Kernel3D.h"
#include <string>
namespace TommyDat {
    class NeuralInput : public NeuralInputBase<Kernel3D<int>> {
    public:
        NeuralInput(std::string path)  {
            data = new Kernel3D<int>(path);
        } ;
    };
}
#endif //RECNN_NEURALINPUT_H
