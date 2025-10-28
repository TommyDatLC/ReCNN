//
// Created by datdau on 10/26/25.
//

#ifndef RECNN_NEURALINPUTBASE_H
#define RECNN_NEURALINPUTBASE_H
namespace TommyDat {
    template <typename Tdata>
    class NeuralInputBase {
    public:
        Tdata* data;
        int lable;
    };
}

#endif //RECNN_NEURALINPUTBASE_H