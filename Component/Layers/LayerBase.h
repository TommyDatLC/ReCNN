//
// Created by datdau on 10/26/25.
//

#ifndef RECNN_LAYERBASE_H
#define RECNN_LAYERBASE_H

#include "../Matrix.h"
#include "../Matrix.h"
#include "../Tracebackable.h"

namespace TommyDat {

    struct Layer {
    private:

        Matrix<Tracebackable<float>>* inActivation = nullptr;
        Matrix<Tracebackable<float>>* outActivation = nullptr;

    public:
        void setOutActivation(Matrix<Tracebackable<float>>* newAc) {
            if (newAc == outActivation)
                return;
            delete outActivation;
            outActivation = newAc;
        }
        Matrix<Tracebackable<float>>* getOutActivation() {
            return outActivation;
        }
        void setInActivation(Matrix<Tracebackable<float>>* newAc) {
            if (newAc == inActivation)
                return;
            delete inActivation;
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