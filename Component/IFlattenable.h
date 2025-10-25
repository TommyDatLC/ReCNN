//
// Created by datdau on 10/25/25.
//

#ifndef RECNN_IFLATTENABLE_H
#define RECNN_IFLATTENABLE_H
namespace TommyDat {
    template <typename T>
    class IFlattenable {
    public:
        virtual T* flatten() = 0;
    };
}

#endif //RECNN_IFLATTENABLE_H