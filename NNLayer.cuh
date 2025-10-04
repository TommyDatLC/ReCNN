//
// Created by datdau on 10/4/25.
//

#ifndef RECNN_NNLAYER_CUH
#define RECNN_NNLAYER_CUH

enum enum_ActivationFunction {
    softmax,
    Tanh,
    relu,
};
class NNLayer {

public:
    enum_ActivationFunction activation_function;
    float* Neuron;
    float* bias;
    int NeuralCount = 0;

    int thisLayerID;
    __device__ void CaculateActivation(NNLayer* lastLayer ,float** matrix) {
        int matrixRow_lastLayerW = lastLayer->NeuralCount;
        int matrxCol_thisW = NeuralCount;

        idx = threadIdx.x + blockDim.x * blockIdx.x;

        for (int i = 0;i < lastLayer->NeuralCount;i++) {
            Neuron[idx] += lastLayer->Neuron[i] * matrix[i][idx];
        }
    }

    void SoftMax() {
        for (int i = 0 ;i < n;i++) {

        }

    }
};


#endif //RECNN_NNLAYER_CUH