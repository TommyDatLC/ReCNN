#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

#include "Component/Layers/ConvolutionLayer.h"
#include "Component/Layers/FClayer.h"
#include "Component/Layers/MaxPoolingLayer.h"
#include "Component/TommyDatNeuralNet/NeuralInput.h"
#include "Component/Serialize.h"


#include "Component/TommyDatNeuralNet/NeuralNetwork.h"
#include  "Utils/File.h"


using namespace std;
namespace fs = std::filesystem;
using namespace TommyDat;

int main() {
    //     Matrix<Tracebackable<float>> a1 = Matrix<Tracebackable<float>>(3,16,16);
    // Matrix<Tracebackable<float>> b1 = Matrix<Tracebackable<float>>(3,16,16);
    // auto test  =a1 - b1;

    // Matrix<float> ker = Matrix<float>(6,3,3);
    // cout << *a.convolution(ker,2);
    // Matrix a = Matrix<float>(1,16,16);
    // auto test  =a.transpose();
    //    cout << a <<  test;
    // cout << a;
    //     cout << "========================================\n";
    //     cout << "   Cat vs Dog CNN Classifier\n";
    //     cout << "========================================\n\n";


    //
    // // ============ LOAD IMAGES ============

    bool useSmallImage = true;  // true = 16x16, false = 400x400
    vector<NeuralInput> trainingData,testData;
    //
    if (useSmallImage) {
        trainingData = ReadImage16x16(false);
        testData = ReadImage16x16(true);
    } else {
      //  trainingData = ReadImage400x400();
    }
    //
    //
    // cout << trainingData.size() << endl;
    //
    // // ============ TEST ============

        auto loadedNet = ModelSerialize::loadNetwork<NeuralInput>("../Models/mymodel.json");

        NeuralNetwork<NeuralInput> net;
        int n = 10;
        net.learningRate = 0.001f;
            globalLearningRate.store(net.learningRate);
        std::thread lrThread(monitorLearningRate);


        auto layer1 = ConvolutionLayer(3, 6, 3, 2);
        auto layer2 = MaxPoolingLayer(2, 2);
        auto fc1 = FClayer(6 * 4 * 4, EnumActivationType::ReLU,true);
        auto fc2 = FClayer( 16, EnumActivationType::ReLU);
        auto output = FClayer(2, EnumActivationType::softMax);
        net.add(&layer1);
        net.add(&layer2);
        net.add(&fc1);
        net.add(&fc2);
        net.add(&output);
        fc1.init();
        fc2.init();
        output.init();
        // Matrix loss(1,1,2,0.f);
        // loss.set(0,0,1,-1);

        // NeuralInput a;
        // a.lable = 1;
        // a.data = new Matrix<Tracebackable<float>>(1,1,10);
        //
        // net.predict(&a);

         for (int i = 0;i < n;i++) {
             net.learningRate = globalLearningRate.load();
            float totalLoss = 0;
             for (int j = 0;j < trainingData.size();j++) {
                // cout << "EPOCH " << i << " " << j << '\n';
                 net.predict(&trainingData[j]);
                 net.backward();

                 if (i % 10 == 0)
                     totalLoss += net.getError();
             }
                 if (totalLoss != 0) {
                     cout << "loss:" << totalLoss << '\n';
                //  auto t = net.getPredictResult();
                // std::cout << "Output Matrix" <<
                //     *(Matrix<float>*)t ;

             }
             //0.666687 0.333313
             //0.686901 0.313099
         }
        lrThread.detach();
        std::string res[] = {"Cat","Dog" };

        cout << "Evaluate model:..";
        // model eval
        Matrix TB(1,outputNeuron,outputNeuron,0);
        for (auto &testImage : testData) {
            net.predict(&testImage);
            int predictID = -1;
            auto ptrvoid_predRes = net.getPredictResult();

            Matrix<float>* ptr_predRes = static_cast<Matrix<float>*>(ptrvoid_predRes);

            int len = ptr_predRes->getLen();
            float* ptr_firstElm = ptr_predRes->flatten();

            float* ptr_maxVal_index = max_element(ptr_firstElm,  ptr_firstElm + len );
            predictID = ptr_maxVal_index - ptr_firstElm;
            cout << "Predict res: " << *ptr_predRes;

            cout << "Training completed, saving model\n";
            ModelSerialize::saveNetwork(net, "../Models/mymodel.json");

            // auto* reloadedNet = ModelSerialize::loadNetwork<float>("../Models/mymodel.json");
            // std::cout << " Model loaded successfully\n";

            // delete reloadedNet;
            return 0;
        }
}