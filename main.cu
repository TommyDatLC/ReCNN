#include <iostream>
#include "Component/Kernel3D.h"
#include "Component/TommyDatNeuralNet/NeuralNetwork.h"
#include "Component/Matrix.h"
#include "Component/Layers/ConvolutionLayer.h"
#include "Component/TommyDatNeuralNet/NeuralInput.h"
using namespace std;
using namespace TommyDat;

int main() {
    Matrix a = Matrix(10,10 ,2.0f);
    a.heInit();
    cout << a;
    cout << a.maxPooling(2,2);
    //Matrix b = Matrix(10,10 ,2.0f);
    // b.set(3,4,-14);
    //kernel.set(1,1,4);
    // NeuralNetwork<NeuralInput> a;
    // auto layer1 = ConvolutionLayer(3,6,10,3);
    // a.Add(&layer1);
    // Kernel3D<float> test(3,3,3);
    //cout  << test;
    return 0;
}
