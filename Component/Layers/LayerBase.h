//
// Created by datdau on 10/26/25.
//

#ifndef RECNN_LAYERBASE_H
#define RECNN_LAYERBASE_H
#include "../Tracebackable.h"

namespace TommyDat {

    class Layer {
    private:
        Matrix<Tracebackable<float>>* activation = nullptr;
    public:
        void setNewActivation(Matrix<Tracebackable<float>>* newAc) {
            delete[] activation;
            activation = newAc;
        }
        Matrix<Tracebackable<float>>* getActivation() {
            return activation;
        }
        void setLastLayer(Layer* newLastLayer) {
            this->lastLayer = newLastLayer;
        }
        void setNextLayer(Layer* newNextLayer) {
            this->nextLayer = newNextLayer;
        }
        //Layer* nextLayer

        virtual void inference(void* lastLayerInput) = 0;
        virtual void backward(void* nextLayerInput) = 0;
    protected:
        Layer* nextLayer = nullptr;
        Layer* lastLayer = nullptr;
    };
}
#endif //RECNN_LAYERBASE_H