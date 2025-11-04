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

        void* inActivation = nullptr;
        void* outActivation = nullptr;

    public:
        void setOutActivation(void* newAc) {
            if (newAc == outActivation)
                return;
            delete outActivation;
            outActivation = newAc;
        }

        void* getOutActivation() {
            return outActivation;
        }
        void setInActivation(void* newAc) {
            if (newAc == inActivation)
                return;
            delete inActivation;
            inActivation = newAc;
        }
        void* getInActivation() {
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