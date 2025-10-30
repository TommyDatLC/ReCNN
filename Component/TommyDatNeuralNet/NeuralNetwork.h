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
        void add(Layer* layer) {
            int n = layers.size();
            if ( n > 0) {
                layer->setLastLayer(layers[n - 1]);
                layers[n - 1]->setNextLayer(layer);
            }
            layers.push_back(layer);
        }
        void predict(TDataInput* input) {
            inputedData = input;
            CheckLayersValid();
            layers[0]->inference( static_cast<void*>( input->data));
        }
        void backward() {
            CheckLayersValid();
            // khởi chạy backward
        }
        Matrix<Tracebackable<float>>* getPredictResult() {
            int n = layers.size();
            return layers[n - 1]->getActivation();
        }
        float CaculateError() {
            Matrix<Tracebackable<float>>* predRes = getPredictResult();
            Matrix copy_predRes = Matrix(*predRes);
            return inputedData->getError(&copy_predRes);
        }
    private:
        TDataInput* inputedData = nullptr;

        void CheckLayersValid() {
            if (layers.size() == 0) {
                throw std::runtime_error("network have no layer to process the operation,please add one");
            }
        }
    };

}
#endif //RECNN_NEURALNETWORK_H