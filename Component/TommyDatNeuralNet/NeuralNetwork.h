//
// Created by datdau on 10/26/25.
//

#ifndef RECNN_NEURALNETWORK_H
#define RECNN_NEURALNETWORK_H
#include <vector>

#include "NeuralInputBase.h"
#include "../Matrix.h"
#include "../Layers/LayerBase.h"

namespace TommyDat {
    template <typename TDataInput>
    class NeuralNetwork {
    public:
        std::vector<Layer*> layers;
        int getSize() {
            return layers.size();
        }
        void Add(Layer* layer) {
            int n = layers.size();
            if ( n > 0) {
                layer->setLastLayer(layers[n - 1]);
                layers[n - 1]->setNextLayer(layer);
            }
            layers.push_back(layer);
        }
        void Predict(TDataInput input) {
            CheckLayersValid();
            layers[0]->inference(input.data);
        }
        void Backward() {
            CheckLayersValid();

        }
        // Matrix<float> GetPredictResult() {
        //
        // }
        void CaculateError() {

        }
    private:
        void CheckLastLayerValid() {

        }
        void CheckLayersValid() {
            if (layers.size() == 0) {
                throw std::runtime_error("network have no layer to process the operation,please add one");
            }
        }
    };

}
#endif //RECNN_NEURALNETWORK_H