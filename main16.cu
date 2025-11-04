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



using namespace std;
namespace fs = std::filesystem;
using namespace TommyDat;

// === HÀM PHỤ ===
bool hasImageExtension(const string& filename) {
    string ext;
    size_t dotPos = filename.find_last_of(".");
    if (dotPos == string::npos) return false;
    ext = filename.substr(dotPos + 1);

    // Chuyển về chữ thường để so sánh
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "jpg" || ext == "jpeg" || ext == "png");
}

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
        NeuralNetwork<NeuralInput> net;
        net.learningRate = 0.01f;
        //
        // // ============ LOAD IMAGES ============

        bool useSmallImage = true;  // true = 16x16, false = 400x400
        vector<NeuralInput> trainingData;
        //
        if (useSmallImage) {
            trainingData = ReadImage16x16();
        } else {
            trainingData = ReadImage400x400();
        }
        //
        //
        // cout << trainingData.size() << endl;
        //
        // // ============ TEST ============

    //     auto loadedNet = ModelSerialize::loadNetwork<NeuralInput>("Models/mymodel.json");
    //
    //     for (auto l : net.layers) {
    //     std::cout << "Layer type: " << typeid(*l).name() << std::endl;
    //     if (auto fc = dynamic_cast<FClayer*>(l)) {
    //         auto W = fc->getWeightMatrix();
    //         auto B = fc->getBiasMatrix();
    //         std::cout << "W dims: " << W->getDim().x << "," << W->getDim().y << "," << W->getDim().z << std::endl;
    //         std::cout << "B dims: " << B->getDim().x << "," << B->getDim().y << "," << B->getDim().z << std::endl;
    //         std::cout << "First 3 weights: " << W->getFlatten(0) << " "
    //                   << W->getFlatten(1) << " "
    //                   << W->getFlatten(2) << std::endl;
    //         }
    //     }
    //
    //     int idx = 0;
    //     for (auto layer : loadedNet->layers) {
    //     std::cout << "Layer #" << idx++ << ": ";
    //
    //     if (auto conv = dynamic_cast<ConvolutionLayer*>(layer)) {
    //         std::cout << "Conv stride=" << conv->getStride();
    //         auto km = conv->getKernelMatrix();
    //         std::cout << ", kernel " << (km ? std::to_string(km->getLen()) : std::string("NULL")) << "\n";
    //     } else if (auto pool = dynamic_cast<MaxPoolingLayer*>(layer)) {
    //         std::cout << "Pool size=" << pool->getSize() << ", stride=" << pool->getStride() << "\n";
    //     } else if (auto fc = dynamic_cast<FClayer*>(layer)) {
    //         std::cout << "FC dense=" << fc->getDense()
    //                   << ", weight=" << (fc->getWeightMatrix() ? std::to_string(fc->getWeightMatrix()->getLen()) : std::string("NULL"))
    //                   << ", bias="  << (fc->getBiasMatrix() ? std::to_string(fc->getBiasMatrix()->getLen()) : std::string("NULL"))
    //                   << "\n";
    //     } else {
    //         std::cout << "Unknown layer\n";
    //     }
    // }

        // // ============ BUILD CNN ============
        cout << "Building CNN architecture...\n";

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
        int n = 50;

        // NeuralInput a;
        // a.lable = 1;
        // a.data = new Matrix<Tracebackable<float>>(1,1,10);
        //
        // net.predict(&a);

         for (int i = 0;i < n;i++) {

            float totalLoss = 0;
             for (int j = 0;j < trainingData.size();j++) {
                // cout << "EPOCH " << i << " " << j << '\n';
                 net.predict(&trainingData[i]);
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


        cout << "Architecture: Input → Conv(3→6) → Pool → FC(96→32) → FC(32→16) → Output(2)\n\n";
        //
        // // ============ TRAINING ============

        // // ============ SAVE THE MODEL AND PREPARE FOR THE NEXT CALL ===========
        cout << "Training completed, saving model\n";

        ModelSerialize::saveNetwork(net, "../Models/mymodel.json");

        auto* loadedNet = ModelSerialize::loadNetwork<float>("../Models/mymodel.json");
        std::cout << " Model loaded successfully\n";

        delete loadedNet;
        return 0;
}