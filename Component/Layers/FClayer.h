// FClayer.h - Fully Connected Layer
// Created by LovelyPoet on 10/27/2025
//

#ifndef RECNN_FCLAYER_H
#define RECNN_FCLAYER_H

#include "LayerBase.h"
#include "../Matrix.h"
#include "../Tracebackable.h"
#include "../EnumActivationType.h"
#include <cmath>
#include <iostream>

namespace TommyDat {
    class FClayer : public Layer {
    private:
        Matrix<float> *WeightMatrix ;
        Matrix<float> *BiasMatrix ;
        Matrix<float> *inputCache = nullptr ;
        Matrix<float> *outputCache = nullptr ;
        EnumActivationType activationType;
        bool _isFirst;
        int _dense;
    public:
        // Constructor
        int getDense() { return _dense; }
        FClayer(int dense, EnumActivationType actType = EnumActivationType::ReLU,bool first = false)
        {
            _dense = dense;
            // He initialization
            _isFirst = first;
            activationType = actType;
          //  std::cout << "FClayer initialized: " << inputSize << " -> " << outputSize << std::endl;
        }
        void init() {
            if (_isFirst) {
                return;
            }
            FClayer* lastFcLayer =  dynamic_cast<FClayer*>(lastLayer );
            WeightMatrix =  new Matrix<float>(1,_dense,lastFcLayer->getDense());
            BiasMatrix = new  Matrix<float>(1, 1, _dense);

         //   std::cout << *BiasMatrix << std::endl;

        }
        // Destructor
        // ~FClayer() {
        //     delete WeightMatrix;
        //     delete BiasMatrix;
        //     if (inputCache) delete inputCache;
        //     if (outputCache) delete outputCache;
        // }


        // Forward pass
        void inference(void* ptr_lastLayerInput) override {

    if (_isFirst) {
        auto tracebackableMaxpooling = static_cast<Matrix<Tracebackable<float>>*>(ptr_lastLayerInput);
        auto floatMaxPoolingOutput = toValueMatrix(*tracebackableMaxpooling);
        if (floatMaxPoolingOutput.getLen() != _dense) {
            throw std::runtime_error("_dense of this layer is not equal to max pooling output size");
        }
        floatMaxPoolingOutput.reShape(1,1,_dense);
       // std::cout << "INPUT FIRST LAYER" << floatMaxPoolingOutput;
        nextLayer->inference(&floatMaxPoolingOutput);
        return;
    }

    Matrix<float>* input = static_cast<Matrix<float> *>(ptr_lastLayerInput);
    if (!input) throw std::runtime_error("FClayer::inference got null input");
    //std::cout << "INPUT HIDDEN LAYER" << *input;
    // optional: ensure weights/bias exist
    if (!WeightMatrix || !BiasMatrix) {
        throw std::runtime_error("FClayer not initialized (WeightMatrix/BiasMatrix null)");
    }

    if (inputCache) { delete inputCache; inputCache = nullptr; }
    inputCache = new Matrix<float>(*input);

    // Z = W * X + b
    auto inputTrans = input->transpose();
    auto out = (*WeightMatrix * inputTrans).transpose() + *BiasMatrix;
  //  std::cout << "FC LAYER " << "WEIGHT " << *WeightMatrix << "INPUTTrans" << inputTrans << "BIAS:" << *BiasMatrix;
    // tính activation TRÊN out (chứ không phải trên outputCache đã set)
    Matrix<float>* activated = nullptr;
    try {
        // tạo bản sao out để áp activation (hoặc modify out rồi wrap)
        activated = new Matrix<float>(out);

        if (activationType == EnumActivationType::ReLU) {
            activated->ReLU();
        } else if (activationType == EnumActivationType::softMax) {
            auto soft = activated->softMax();
            delete activated;
            activated = new Matrix<float>(soft);
        }
        // khác: giữ nguyên (no activation)

        // bây giờ gán outputCache (xóa cũ) và setOutActivation
        outputCache = activated; // chuyển ownership

        // setOutActivation nên nhận pointer tới output sau activation
        setOutActivation(outputCache);

        // Forward to next layer
        if (nextLayer != nullptr) {
            nextLayer->inference(outputCache);
        }

    } catch (...) {
        // Nếu có exception trong quá trình tạo/activation -> dọn dẹp
        if (activated) delete activated;
        throw; // rethrow để caller biết
    }
}


        // Backward pass
        void backward(void* ptr_nextLayerGradient, float learningRate = 0.0001f) override {
            Matrix<float>* dA_next = static_cast<Matrix<float>*>(ptr_nextLayerGradient);
            if (lastLayer == nullptr) {
                // No previous layer to propagate to
                return;
            }
            if (_isFirst) {
                lastLayer->backward(ptr_nextLayerGradient,learningRate);
                return;
            }
            // --- 1) compute dZ from dA_next depending on activation ---
            Matrix<float>* dZ = nullptr;

            if (activationType == EnumActivationType::softMax) {
                // With softmax + cross-entropy, dZ = dA_next (assumed)
                dZ = new Matrix<float>(*dA_next);

            } else if (activationType == EnumActivationType::ReLU) {
                // derivative of ReLU: 1 where outputCache > 0 else 0
                if (!outputCache) throw std::runtime_error("outputCache is null in backward ReLU");
                // make a copy of outputCache to compute derivative
                Matrix<float> dAct(*outputCache); // copy
                // dAct = dAct > 0 ? 1 : 0
                dAct.apply([] __device__ (float x) {
                    return (x > 0.0f) ? 1.0f : 0.0f;
                });
                // elementwise multiply dA_next * dAct
                dZ = mulUnofficial<float, float>( *dA_next, dAct ); // returns new Matrix<float>*

            } else {
                // no activation or unsupported: assume dZ = dA_next
                dZ = new Matrix<float>(*dA_next);
            }

            // --- 2) prepare previous activation (A_prev) ---
            Matrix<float>* A_prev = inputCache;


            // --- 3) compute gradients ---
            // shapes recap (our convention in inference):
            // A_prev: 1 x 1 x lastDim  (n=1, m=lastDim)
            // dZ:     1 x 1 x dense    (n=1, m=dense)
            // We want:
            // dW = dZ_transpose * A_prev   -> shape 1 x dense x lastDim
            // dB = sum over examples of dZ -> shape 1 x 1 x dense
            // dA_prev = (W^T * dZ_transpose).transpose() -> shape 1 x 1 x lastDim

            // transpose dZ to shape 1 x dense x 1
            Matrix<float> dZ_t = dZ->transpose(); // temporary by-value
            // compute dW = dZ_t * A_prev  (returns Matrix by value)
            Matrix<float> dW = dZ_t * (*A_prev); // result is 1 x dense x lastDim

            // If you want to average by number of examples (batch size), compute mExamples.
            // Here we assume batch size = A_prev.m (or 1 if shaped like 1x1xN). We'll detect examples = A_prev->getLen() / lastDim?
            // Because typical usage here is single-example, but to be safe try to infer "mExamples"
            int mExamples = 1;
            // attempt to infer: if A_prev has shape (1, n, m) and n==1 then m is number of features, not batch.
            // Without explicit batching convention we'll leave as 1. If you batch externally, adapt here.
            if (mExamples > 1) {
                // divide dW by mExamples
                dW.apply([mExamples] __device__ (float x) {
                    return x / (float)mExamples;
                });
            }

            // dB: sum dZ along rows (axis N) to get shape 1 x 1 x dense
            Matrix<float>* dB = dZ->sumAlongAxis(AXIS_N); // returns new Matrix<float>*

            // scale updates by learning rate
            dW.apply([learningRate] __device__ (float x) {
                return x * learningRate;
            });
            dB->apply([learningRate] __device__ (float x) {
                return x * learningRate;
            });

            // --- 4) Update parameters: W = W - dW; b = b - dB ---
            // compute new weights and bias
            Matrix<float> newW = (*WeightMatrix) - dW; // operator- returns Matrix
            Matrix<float> newB = (*BiasMatrix) - (*dB);

            // replace old pointers (free old)
            delete WeightMatrix;
            delete BiasMatrix;
            WeightMatrix = new Matrix<float>(newW);
            BiasMatrix = new Matrix<float>(newB);

            // --- 5) compute dA_prev to pass to lastLayer ---
            // W^T:
            Matrix<float> W_t = WeightMatrix->transpose(); // 1 x lastDim x dense
            // dZ transpose again to shape 1 x dense x 1 (we already had dZ_t)
            // multiply: W_t * dZ_t  -> 1 x lastDim x 1
            Matrix<float> dA_prev_temp = W_t * dZ_t; // 1 x lastDim x 1
            // transpose to shape 1 x 1 x lastDim to match activation shape
            Matrix<float> dA_prev = dA_prev_temp.transpose();

            // propagate to last layer
           lastLayer->backward(&dA_prev, learningRate);
          // std::cout << "A_prev" << *A_prev << "DZ" <<  *dZ << "LAYER BIAS CHANGE" << *dB << "LAYER WEIGHT CHANGE " << dW ;

            // cleanup temporaries
            delete dZ;
            delete dB;
            // note: dA_prev, dA_prev_temp, W_t, dW, dZ_t are stack/value objects and will be freed by their destructors
        }

        // Get activation output (theo LayerBase)
        Matrix<float>* getOutActivation() {
            return outputCache;
        }

        // Print weights (for debugging)
        void printWeight() {

            std::cout << " FClayer Weights " << std::endl;
            std::cout << WeightMatrix;
            std::cout << "Bias:" << std::endl;
            std::cout << BiasMatrix;
        }
        void CheckWeight(int i ) {
            if (WeightMatrix == NULL)
                std::cout << "weight matrix is null " << i << '\n';
        }
    };
}

#endif // RECNN_FCLAYER_H
