//
// Created by datdau on 10/26/25.
//

#ifndef RECNN_LAYERBASE_H
#define RECNN_LAYERBASE_H
namespace TommyDat {

    class Layer {
    public:
        void setLastLayer(Layer* lastLayer) {
            this->lastLayer = lastLayer;
        }
        void setNextLayer(Layer* lastLayer) {
            this->nextLayer = nextLayer;
        }
        //,Layer* nextLayer
        virtual void inference(void* lastLayerInput) = 0;
        virtual void backward(void* nextLayerInput) = 0;
    protected:
        Layer* nextLayer;
        Layer* lastLayer;
    };
}
#endif //RECNN_LAYERBASE_H