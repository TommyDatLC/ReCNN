// //
// // Created by LovelyPoet on 10/27/2025
// //
// #ifndef RECNN_NNLAYER_H
// #define RECNN_NNLAYER_H
//
// #include "LayerBase.h"
// #include "Matrix.h"
// #include "../Component/TommyDatMatrix.h"
// #include "../Component/EnumActivation.h"
//
// namespace TommyDat {
// class NNlayer : public Layer {
//
// private :
// // all the main matrix
// Matrix<float> WeightMatrix; //W : weight
// Matrix<float> BiasMatrix ; // b : bias
// Matrix<float> inputCache; // input saved for backward
// Matrix<float> outputCache; // output saved for activation
// EnumActivationType activationType;
//
// // activation function
// Matrix<float> activate(const Matrix<float>& Z) {
//         switch (activationType) {
//              case EnumActivationType: :RELU:
//                   return Z.apply([](float x) { return x > 0 ? x : 0; });
//              case EnumActivationType: :SIGMOID:
//                   return Z.apply([](float x) { return 1.0f / (1.0f + expf(-x)); });
//              case EnumActivationType: :TANH:
//                   return Z.apply([](float x) { return tanhf(x); } );
//
//              default:
//                   return Z;
//   }
//  }
// // gradient descent for backward, backpropagation
// Matrix<float> activateDerivative(const Matrix<float>& Z) {
//        switch (activationType) {
//              case EnumActivationType: :RELU:
//                   return Z.apply([](float x) { return x > 0 ? 1.0f  : 0.0f; });
//              case EnumActivationType: :SIGMOID:
//                   Matrix<float> s = Z.apply([](float x) { return 1.0f / (1.0f + expf(-x)); });
//                   return s * (1.0f - s);
//              case EnumActivationType: :TANH:
//                   return Z.apply([](float x) { float t = tanhf(x); return 1.0f - t* t ; } );
//
//              default:
//                   return Matrix<float>(Z.getDim());
//   }
//  }
// // Layer Constructor
// NNlayer(int inputSize, int outputSize, EnumActivationType actType) {
//         activationType = actType;
//         WeightMatrix = Matrix<float>::random(outputSize, inputSize); // for random W
//         BiasMatrix = Matrix<float>::zeros(outputSize, 1); //zero bias
//  }
//
// //forward
// void inference(void* ptr_lastLayerInput) override {
//      Matrix<float>* input= static_cast<Matrix<float>*>(ptr_lastLayerInput);
//      inputCache = *input;
//
//      Matrix<float> Z = WeightMatrix * (*input) + BiasMatrix;
//      Matrix<float> A = activate(Z);
//      outputCache = A;
//
//      if (nextLayer != nullptr)
//         nextLayer->inference(&A);
//  }
//
// //Backward
// void backward(void* ptr_nextLayerInput) override {
//      Matrix<float>* dA_next = static_cast<Matrix<float>*>(ptr_nextLayerInput); // gradient from next layer
//
//      Matrix<float> dZ = (*dA_next) * activateDerivate(outputCache); // compute gradient with respect to the maintenance output
//      Matrix<float> dW = dZ * inputCache.transpose(); // gradient of weight
//      Matrix<float> db = dZ.sumAlongAxis(1); // gradient of bias
//
//      if (lastLayer != nullptr) {
//         Matrix<float> dA_prev = WeightMatrix.transpose() * dZ; // gradient backward for  the before layer
//         lastLayer->backward(&dA_prev);
//      }
//
//      float lr = 0.01f;
//      WeightMatrix = WeightMatrix - lr * dW;
//      BiasMatrix = BiasMatrix - lr * db;
//  }
//
// }
//
