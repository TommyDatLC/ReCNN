#include <iostream>

#include "Component/TommyDatNeuralNet/NeuralNetwork.h"
#include "Component/Matrix.h"
#include "Component/Layers/ConvolutionLayer.h"
#include "Component/Layers/MaxPoolingLayer.h"

#include "Component/TommyDatNeuralNet/NeuralInput.h"
using namespace std;
using namespace TommyDat;

int main() {
    Matrix a = Matrix(10,3,3 ,2.0f);
    Matrix b = Matrix(3,5,5,10.f);


    a.heInit(); // heInit is random on neural network initalizeation
    cout << a;
    a.normalize();
      a.apply([] __device__ (int x) {
          return x * x;
      });
    a.transpose();

    cout << "after normalize \n" << a;
     NeuralNetwork<NeuralInput> net;
    auto layer1 = ConvolutionLayer(3,6,3,3);
    auto layer2 = MaxPoolingLayer(2,2);
    net.add(&layer1);
    net.add(&layer2);
    NeuralInput n =  NeuralInput("./testdata1.png");
    net.predict(n);
    net.GetPredictResult();
    return 0;
}
