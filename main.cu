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
    //
    // Matrix a = Matrix<float>(1,10,10,-0.000000000001f);
    // a.set(0,0,0,1);
    // cout << a;
    //
    //  cout << "before:\n" << a << '\n';
    //
    // cout << "after:\n" << *a.maxPooling(2,2);
    // cout << "input ker:\n" << ker;
    // cout << "output \n" << *a.convolution(ker);

    NeuralNetwork<NeuralInput> net;
    auto layer1 = ConvolutionLayer(3,6,3,2);
    auto layer2 = MaxPoolingLayer(2,2);
    net.add(&layer1);
    net.add(&layer2);
    NeuralInput n = NeuralInput("./testdata1.png");
    net.predict(&n);
    auto predictRes = net.getPredictResult();
    float err = net.CaculateError();
    cout << err;
    // return 0;
}
