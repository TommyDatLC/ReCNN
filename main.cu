#include <iostream>
#include "Component/Matrix.h"
#include "Component/Layers/ConvolutionLayer.h"
#include "Component/Layers/MaxPoolingLayer.h"
#include "Component/TommyDatNeuralNet/NeuralInput.h"
#include "Component/TommyDatNeuralNet/NeuralNetwork.h"


using namespace std;
using namespace TommyDat;


// make a class to backtracing
// checking the he init why always random
int main() {

    //
    // Matrix ker = Matrix<float>(6,3,3,1);
    // Matrix a = Matrix<float>(3,5,5,0.2);
    // auto out = a.convolution(ker,1);
    // cout << *out;
    //
    //     cout <<  (Tracebackable<float>(4) += 4);
    // cout << a;
    //
    //  cout << "before:\n" << a << '\n';
    //
    // cout << "after:\n" << *a.maxPooling(2,2);
    // cout << "input ker:\n" << ker;
    // cout << "output \n" << *a.convolution(ker);

    NeuralNetwork<NeuralInput> net;
    net.learningRate = 0.01f;
    auto layer1 = ConvolutionLayer(3,6,3,2);
    auto layer2 = MaxPoolingLayer(2,2);
    net.add(&layer1);
    net.add(&layer2);
    NeuralInput n = NeuralInput("./testdata1.png");
    net.predict(&n);
    auto predictRes = net.getPredictResult();
    net.backward();
    return 0;
}
