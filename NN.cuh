//
// Created by datdau on 10/4/25.
//

#ifndef RECNN_NN_CUH
#define RECNN_NN_CUH
#include <signal.h>
#include <stdexcept>

#include "NNLayer.cuh"


class NN {
public:
    int maxLayer = 0;
    NN(int maxLayer) {
        this->maxLayer = maxLayer;
        layers = new NNLayer[maxLayer];
    }
    // void ModifyLayer(int id,float* NeuralActivation,float* bias,enum_ActivationFunction activation_function) {
    //
    //     int NeuralCount = sizeof(NeuralActivation) / sizeof(float);
    //     int biasCount = sizeof(NeuralActivation) / sizeof(float);
    //     if (biasCount != NeuralCount) {
    //         throw std::__throw_runtime_error("Số neuron khác số bias");
    //     }
    //
    //     layers[id] = NNLayer();
    //     layers[id].activation_function = activation_function;
    //     layers[id].Neuron = NeuralActivation;
    //     layers[id].bias = bias;
    //     ModifyWeight(id,sizeof(NeuralActivation) / sizeof(float));
    // }
    void ModifyLayer(int id,int NeuralCount,enum_ActivationFunction activation_function) {
        layers[id] = NNLayer();
        layers[id].activation_function = activation_function;
        layers[id].Neuron = new float[NeuralCount] {};
        layers[id].bias = new float[NeuralCount] {};
        layers[id].NeuralCount = NeuralCount;
        ModifyWeight(id,NeuralCount);
    }
    template <size_t n>
    void Forward(float (&input)[n]) {
        if (n != layers[0].NeuralCount)
            throw std::runtime_error("Số neuron trong neural network không khớp với input");
        layers[0].Neuron = input;
        for (int i = 0 ;i < n;i++) {

        }

    }
    void GetResult()
    {

    }

private:
    NNLayer* layers;
    float*** _weights; // layers lastlayerWeight thisLayerWeight
    void ModifyWeight(int id,int count) {
        int lastLayerCount = layers[id - 1].NeuralCount;
        _weights[id] = new float*[lastLayerCount];


        for (int i = 0 ;i <  lastLayerCount;i++) {
            _weights[id][i] = new float[count];
            for (int j = 0;j < count;j++) {
                _weights[id][i][j] = rand();
            }
        }
        int nextLayerCount = layers[id + 1].NeuralCount;
        for (int i = 0 ;i <  count;i++) {
            _weights[id + 1][i] = new float[count];
            for (int j = 0;j < nextLayerCount ;j++) {
                _weights[id + 1][i][j] = rand();
            }
        }

    }

};


#endif //RECNN_NN_CUH