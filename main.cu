#include <iostream>
#include "Component/Kernel3D.h"
#include "Component/TommyDatNeuralNet/NeuralNetwork.h"
#include "Component/Matrix.h"
#include "Component/Layers/ConvolutionLayer.h"
#include "Component/Layers/MaxPoolingLayer.h"
#include "Component/TommyDatNeuralNet/NeuralInput.h"
using namespace std;
using namespace TommyDat;

int main() {
    // Matrix a = Matrix(10,10 ,2.0f);
    // a.heInit();
    // cout << a;
    // cout << a.maxPooling(2,2);
    //
    // Matrix b = Matrix(10,10 ,2.0f);
    //  b.set(3,4,-14);
    // kernel.set(1,1,4);
     NeuralNetwork<NeuralInput> net;
    NeuralInput testInput = NeuralInput("/home/tommydatlc/Pictures/Screenshots/Screenshot From 2025-10-08 17-41-23.png");
     auto layer1 = ConvolutionLayer(3,6,3,1);
    auto maxPoolingLayer = MaxPoolingLayer(2,2);
     net.add(&layer1);
     net.add(&maxPoolingLayer);
     net.predict(testInput);
    //  Kernel3D<float> test(3,3,3);
    // cout  << test;
    return 0;
}
