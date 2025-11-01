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
    // Matrix ker = Matrix<float>(10,32,32,1);
    // ker.set(0,1,1,4);
    // cout << *ker.softMax() << '\n';

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
    NeuralInput data = NeuralInput("./testdata1.png");
    data.lable = 5;
    net.learningRate = 1.f;
    auto layer1 = ConvolutionLayer(3,6,3,2);
    auto layer2 = MaxPoolingLayer(2,2);
    net.add(&layer1);
    net.add(&layer2);
    int epoc = 1000;

    for (int i = 0;i < epoc;i++) {
        //cout << "ep: " << i << '\n';

        net.predict(&data);
      //  auto predictRes = net.getPredictResult();
        net.backward();
        if (i % 100 == 0) {
             auto test = net.getPredictResult();
             cout <<"test result:" << i<< '\n'  << test->getFlatten( data.lable)<< *test << '\n';
            // cout << net.CaculateError() << '\n';
            //layer1.printWeight();
        }

    }



}
