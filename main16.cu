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

// === HÀM PHỤ ===

// === HÀM ĐỌC THƯ MỤC ẢNH ===
vector<NeuralInput> ReadImageFolder(const string& folderPath, int label) {
    vector<NeuralInput> res;

    if (!fs::exists(folderPath)) {
        cerr << "Folder not found: " << folderPath << endl;
        return res;
    }

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            string path = entry.path().string();
            if (hasImageExtension(path)) {
                try {
                    NeuralInput a(path);
                    a.lable = label;
                    res.push_back(a);
                } catch (const exception& e) {
                    cerr << "Error loading " << path << ": " << e.what() << endl;
                }
            }
        }
    }

    return res;
}

// === HÀM ĐỌC ẢNH 16x16 ===
vector<NeuralInput> ReadImage16x16() {
    vector<NeuralInput> res;

    cout << "Loading 16x16 images...\n";

    string catPath = "./Dataset/cat/16x16";
    string dogPath = "./Dataset/dog/16x16";

    vector<NeuralInput> cats = ReadImageFolder(catPath, 0);
    vector<NeuralInput> dogs = ReadImageFolder(dogPath, 1);

    res.insert(res.end(), cats.begin(), cats.end());
    res.insert(res.end(), dogs.begin(), dogs.end());

    cout << "Loaded " << res.size() << " images (16x16): "
         << cats.size() << " cats, " << dogs.size() << " dogs\n";

    return res;
}

// === HÀM ĐỌC ẢNH 400x400 ===
vector<NeuralInput> ReadImage400x400() {
    vector<NeuralInput> res;

    cout << "Loading 400x400 images...\n";

    string catPath = "./Dataset/cat/400x400";
    string dogPath = "./Dataset/dog/400x400";

    vector<NeuralInput> cats = ReadImageFolder(catPath, 0);
    vector<NeuralInput> dogs = ReadImageFolder(dogPath, 1);

    res.insert(res.end(), cats.begin(), cats.end());
    res.insert(res.end(), dogs.begin(), dogs.end());

    cout << "Loaded " << res.size() << " images (400x400): "
         << cats.size() << " cats, " << dogs.size() << " dogs\n";

    return res;
}

// ============================================
// MAIN
// ============================================
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
         //   trainingData = ReadImage400x400();
        }

        //
        //
        // cout << trainingData.size() << endl;
        //
        // // ============ BUILD CNN ============
        // cout << "Building CNN architecture...\n";
        //
        NeuralNetwork<NeuralInput> net;
        int n = 10;
        net.learningRate = 0.001f;
            globalLearningRate.store(net.learningRate);
        std::thread lrThread(monitorLearningRate);

        int outputNeuron = 2;
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

            int t = TB.get(0,testImage.lable,predictID);
            TB.set(0,testImage.lable,predictID,t+1);
        }
        cout << "truth table"<< TB;
        int sum = 0;
        for (int i = 0;i < outputNeuron;i++) {
            sum += TB.get(0,i,i);
        }
        cout << "accuricy:" << sum / testData.size();
        //
        // // ============ TRAINING ============


        auto* reloadedNet = ModelSerialize::loadNetwork<float>("../Models/mymodel.json");

        std::cout << " Model loaded successfully\n";

        delete reloadedNet;
        return 0;

}