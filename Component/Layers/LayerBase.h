//
// Created by datdau on 10/26/25.
//

#ifndef RECNN_LAYERBASE_H
#define RECNN_LAYERBASE_H
#include "../Tracebackable.h"

namespace TommyDat {

    class Layer {
    private:
        Matrix<Tracebackable<float>>* inActivation = nullptr;
        Matrix<Tracebackable<float>>* outActivation = nullptr;
    public:
        void setOutActivation(Matrix<Tracebackable<float>>* newAc) {
            delete[] outActivation;
            outActivation = newAc;
        }
        Matrix<Tracebackable<float>>* getOutActivation() {
            return outActivation;
        }
        void setInActivation(Matrix<Tracebackable<float>>* newAc) {
            delete[] inActivation;
            inActivation = newAc;
        }
        Matrix<Tracebackable<float>>* getInActivation() {
            return inActivation;
        }
        void setLastLayer(Layer* newLastLayer) {
            this->lastLayer = newLastLayer;
        }
        void setNextLayer(Layer* newNextLayer) {
            this->nextLayer = newNextLayer;
        }
        //Layer* nextLayer

        virtual void inference(void* lastLayerInput) = 0;
        virtual void backward(void* nextLayerInput,float learningRate) = 0;
    protected:
        Layer* nextLayer = nullptr;
        Layer* lastLayer = nullptr;
    };
}
#endif //RECNN_LAYERBASE_H