#include <iostream>

#include "Component/TommyDatNeuralNet/NeuralNetwork.h"
#include "Component/Matrix.h"
#include "Component/Layers/ConvolutionLayer.h"
#include "Component/Layers/MaxPoolingLayer.h"

#include "Component/TommyDatNeuralNet/NeuralInput.h"
using namespace std;
using namespace TommyDat;


// make a class to backtracing
// checking the he init why always random
int main() {
    //  Matrix ker = Matrix<Tracebackable<float>>(9,3,3);
    // Matrix a = Matrix<Tracebackable<float>>(3,32,32,3.f);
    // a.normalize();
    // cout << "input ker:\n" << ker;
    // cout << "output \n" << *a.convolution(ker);

     NeuralNetwork<NeuralInput> net;
    auto layer1 = ConvolutionLayer(3,6,3,2);
    auto layer2 = MaxPoolingLayer(2,2);
    net.add(&layer1);
    net.add(&layer2);
    NeuralInput n = NeuralInput("./testdata1.png");

    net.predict(n);
   // net.GetPredictResult();
    return 0;
}
