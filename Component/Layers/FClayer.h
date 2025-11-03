//
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
        Matrix<float> *inputCache ;
        Matrix<float> *outputCache ;
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

        Matrix<float>* getWeightMatrix() const { return WeightMatrix; }
        Matrix<float>* getBiasMatrix() const { return BiasMatrix; }
        EnumActivationType getActivationType() const { return activationType; }
        bool isFirst() const { return _isFirst; }

        void setWeightMatrix(const Matrix<float>& mat) {
            if (WeightMatrix) delete WeightMatrix;
            WeightMatrix = new Matrix<float>(mat);
        }

        void setBiasMatrix(const Matrix<float>& mat) {
            if (BiasMatrix) delete BiasMatrix;
            BiasMatrix = new Matrix<float>(mat);
        }


        void init() {
            if (_isFirst) {
                return;
            }
            FClayer* lastFcLayer =  dynamic_cast<FClayer*>(lastLayer );
            WeightMatrix =  new Matrix<float>(1,_dense,lastFcLayer->getDense());
            BiasMatrix = new  Matrix<float>(1, 1, _dense);

            std::cout << *BiasMatrix << std::endl;

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
                auto floatMaxPoolingOutput =  toValueMatrix( *tracebackableMaxpooling);
                if (floatMaxPoolingOutput.getLen() != _dense) {
                    throw std::runtime_error("_dense of this layer is not equal to max pooling output size");
                }
                floatMaxPoolingOutput.reShape(1,1,_dense);
                nextLayer->inference(&floatMaxPoolingOutput);

                return;
            }
            // else if it not the first
            // Get input from previous layer or initial input
            Matrix<float>* input = static_cast<Matrix<float> *>(ptr_lastLayerInput);


            // Cache input for backward pas
            // Z = W * X + b
            auto inputTrans =  input->transpose();
            auto out = (*WeightMatrix * inputTrans).transpose() + *BiasMatrix  ;
            std::cout << "weight" << *WeightMatrix << "inputTRan" << inputTrans << "out" << out << std::endl;
            outputCache = new Matrix<float>(out );

            // Apply activation
            if (activationType == EnumActivationType::ReLU) {
                outputCache->ReLU();

            }
            if (activationType == EnumActivationType::softMax) {
                // Apply softmax
                auto softMaxResult = outputCache->softMax();

                outputCache = new Matrix<float>( softMaxResult);
            }
            // Else: no activation
          //  std::cout << outputCache << std::endl;
            // Forward to next layer
            if (nextLayer != nullptr) {
                nextLayer->inference(outputCache);
            }
        }

        // Backward pass
        void backward(void* ptr_nextLayerGradient, float learningRate = 0.01f) override {
            // Matrix<float>* dA_next = static_cast<Matrix<float>*>(ptr_nextLayerGradient);
            //
            // // Compute activation derivative
            // Matrix<float> outputValues = (*outputCache);
            // Matrix<float>* activation_grad = nullptr;
            //
            // if (activationType == EnumActivationType::ReLU) {
            //     activation_grad = new Matrix<float>(outputValues);
            //     activation_grad->apply([] __device__ (float x) { return x > 0; });
            // } else if (activationType == EnumActivationType::softMax) {
            //     // For softmax + cross-entropy, gradient is already computed correctly
            //     // Just pass through (derivative = 1)
            //     dim3 dim = outputCache->getDim();
            //     activation_grad = new Matrix<float>(dim, 1.0f);
            // } else {
            //     // No activation: derivative = 1
            //     dim3 dim = outputCache->getDim();
            //     activation_grad = new Matrix<float>(dim, 1.0f);
            // }
            //
            // // dZ = dA * activation_derivative
            // auto dZ = mulUnofficial(*dA_next, *activation_grad);
            // delete activation_grad;
            //
            // // dW = dZ * input^T
            // auto inputT = inputCache->transpose();
            // Matrix<float> inputT_values = (*inputT);
            // auto dW = mulUnofficial(*dZ, inputT_values);
            // delete inputT;
            //
            // // db = sum(dZ) along axis 1 (rows)
            // auto db = SumAlongAxis(*dZ, 1);
            //
            // // Update weights: W -= lr * dW
            // // Convert Matrix<float>* dW to proper type
            // dim3 dW_dim = dW->getDim();
            // Matrix<Tracebackable<float>>* dW_trace = new Matrix<Tracebackable<float>>(dW_dim);
            // for (int i = 0; i < dW->getLen(); i++) {
            //     dW_trace->setFlatten(i, Tracebackable<float>(dW->getFlatten(i) * learningRate));
            // }
            //
            // auto newWeight = *WeightMatrix - *dW_trace;
            // delete WeightMatrix;
            // WeightMatrix = newWeight;
            // delete dW;
            // delete dW_trace;
            //
            // // Update bias: b -= lr * db
            // dim3 db_dim = db->getDim();
            // Matrix<Tracebackable<float>>* db_trace = new Matrix<Tracebackable<float>>(db_dim);
            // for (int i = 0; i < db->getLen(); i++) {
            //     db_trace->setFlatten(i, Tracebackable<float>(db->getFlatten(i) * learningRate));
            // }
            //
            // auto newBias = *BiasMatrix - *db_trace;
            // delete BiasMatrix;
            // BiasMatrix = newBias;
            // delete db;
            // delete db_trace;
            //
            // // Propagate gradient to previous layer
            // if (lastLayer != nullptr) {
            //     auto WT = WeightMatrix->transpose();
            //     Matrix<float> WT_values = (*WT);
            //     auto dA_prev = mulUnofficial(WT_values, *dZ);
            //     delete WT;
            //
            //     lastLayer->backward(dA_prev, learningRate);
            //     delete dA_prev;
            // }
            //
            // delete dZ;
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
    };
}

#endif // RECNN_FCLAYER_H