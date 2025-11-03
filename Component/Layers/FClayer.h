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
        Matrix<Tracebackable<float>>* WeightMatrix;
        Matrix<Tracebackable<float>>* BiasMatrix;
        Matrix<Tracebackable<float>>* inputCache;
        Matrix<Tracebackable<float>>* outputCache;
        EnumActivationType activationType;

    public:
        // Constructor
        FClayer(int inputSize, int outputSize, EnumActivationType actType = EnumActivationType::ReLU)
            : activationType(actType), inputCache(nullptr), outputCache(nullptr) {

            // He initialization
            float stddev = sqrtf(2.0f / inputSize);

            WeightMatrix = new Matrix<Tracebackable<float>>(1, outputSize, inputSize);
            WeightMatrix->heInit(inputSize);

            BiasMatrix = new Matrix<Tracebackable<float>>(1, outputSize, 1, 0.0f);

            std::cout << "FClayer initialized: " << inputSize << " -> " << outputSize << std::endl;
        }

        // Destructor
        ~FClayer() {
            delete WeightMatrix;
            delete BiasMatrix;
            if (inputCache) delete inputCache;
            if (outputCache) delete outputCache;
        }

        // Forward pass
        void inference(void* ptr_lastLayerInput) override {
            // Get input from previous layer or initial input
            Matrix<Tracebackable<float>>* input = nullptr;

            if (lastLayer != nullptr) {
                input = lastLayer->getOutActivation();
            } else {
                // Input từ NeuralInput->data
                input = static_cast<Matrix<Tracebackable<float>>*>(ptr_lastLayerInput);
            }

            // Cache input for backward pass
            if (inputCache) delete inputCache;
            inputCache = new Matrix<Tracebackable<float>>(*input);

            // Z = W * X + b
            auto weighted = *WeightMatrix * *input;

            // Clean old output
            if (outputCache) delete outputCache;

            // Add bias
            outputCache = *weighted + *BiasMatrix;
            delete weighted;

            // Apply activation
            if (activationType == EnumActivationType::ReLU) {
                outputCache->ReLU();
            } else if (activationType == EnumActivationType::softMax) {
                // Apply softmax
                auto softMaxResult = outputCache->softMax();
                delete outputCache;

                // Convert back to Tracebackable
                dim3 dim = softMaxResult->getDim();
                outputCache = new Matrix<Tracebackable<float>>(dim);
                for (int i = 0; i < softMaxResult->getLen(); i++) {
                    outputCache->setFlatten(i, Tracebackable<float>(softMaxResult->getFlatten(i)));
                }
                delete softMaxResult;
            }
            // Else: no activation

            // Forward to next layer
            if (nextLayer != nullptr) {
                nextLayer->inference(outputCache);
            }
        }

        // Backward pass
        void backward(void* ptr_nextLayerGradient, float learningRate = 0.01f) override {
            Matrix<float>* dA_next = static_cast<Matrix<float>*>(ptr_nextLayerGradient);

            // Compute activation derivative
            Matrix<float> outputValues = toValueMatrix<float>(*outputCache);
            Matrix<float>* activation_grad = nullptr;

            if (activationType == EnumActivationType::ReLU) {
                activation_grad = new Matrix<float>(outputValues);
                activation_grad->apply([](float x) { return x > 0 ? 1.0f : 0.0f; });
            } else if (activationType == EnumActivationType::softMax) {
                // For softmax + cross-entropy, gradient is already computed correctly
                // Just pass through (derivative = 1)
                dim3 dim = outputCache->getDim();
                activation_grad = new Matrix<float>(dim, 1.0f);
            } else {
                // No activation: derivative = 1
                dim3 dim = outputCache->getDim();
                activation_grad = new Matrix<float>(dim, 1.0f);
            }

            // dZ = dA * activation_derivative
            auto dZ = mulUnofficial(*dA_next, *activation_grad);
            delete activation_grad;

            // dW = dZ * input^T
            auto inputT = inputCache->transpose();
            Matrix<float> inputT_values = toValueMatrix<float>(*inputT);
            auto dW = mulUnofficial(*dZ, inputT_values);
            delete inputT;

            // db = sum(dZ) along axis 1 (rows)
            auto db = SumAlongAxis(*dZ, 1);

            // Update weights: W -= lr * dW
            // Convert Matrix<float>* dW to proper type
            dim3 dW_dim = dW->getDim();
            Matrix<Tracebackable<float>>* dW_trace = new Matrix<Tracebackable<float>>(dW_dim);
            for (int i = 0; i < dW->getLen(); i++) {
                dW_trace->setFlatten(i, Tracebackable<float>(dW->getFlatten(i) * learningRate));
            }

            auto newWeight = *WeightMatrix - *dW_trace;
            delete WeightMatrix;
            WeightMatrix = newWeight;
            delete dW;
            delete dW_trace;

            // Update bias: b -= lr * db
            dim3 db_dim = db->getDim();
            Matrix<Tracebackable<float>>* db_trace = new Matrix<Tracebackable<float>>(db_dim);
            for (int i = 0; i < db->getLen(); i++) {
                db_trace->setFlatten(i, Tracebackable<float>(db->getFlatten(i) * learningRate));
            }

            auto newBias = *BiasMatrix - *db_trace;
            delete BiasMatrix;
            BiasMatrix = newBias;
            delete db;
            delete db_trace;

            // Propagate gradient to previous layer
            if (lastLayer != nullptr) {
                auto WT = WeightMatrix->transpose();
                Matrix<float> WT_values = toValueMatrix<float>(*WT);
                auto dA_prev = mulUnofficial(WT_values, *dZ);
                delete WT;

                lastLayer->backward(dA_prev, learningRate);
                delete dA_prev;
            }

            delete dZ;
        }

        // Get activation output (theo LayerBase)
        Matrix<Tracebackable<float>>* getOutActivation() {
            return outputCache;
        }

        // Print weights (for debugging)
        void printWeight() {
            std::cout << " FClayer Weights " << std::endl;
            std::cout << *WeightMatrix;
            std::cout << "Bias:" << std::endl;
            std::cout << *BiasMatrix;
        }
    };
}

#endif // RECNN_FCLAYER_H