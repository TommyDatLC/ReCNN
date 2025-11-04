#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <random>
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

// // === HÀM ĐỌC ẢNH 16x16 ===
// vector<NeuralInput> ReadImage16x16() {
//     vector<NeuralInput> res;
//
//     cout << "Loading 16x16 images...\n";
//
//     string catPath = "./Dataset/cat/16x16";
//     string dogPath = "./Dataset/dog/16x16";
//
//     vector<NeuralInput> cats = ReadImageFolder(catPath, 0);
//     vector<NeuralInput> dogs = ReadImageFolder(dogPath, 1);
//
//     res.insert(res.end(), cats.begin(), cats.end());
//     res.insert(res.end(), dogs.begin(), dogs.end());
//
//     cout << "Loaded " << res.size() << " images (16x16): "
//          << cats.size() << " cats, " << dogs.size() << " dogs\n";
//
//     return res;
// }
//
// // === HÀM ĐỌC ẢNH 400x400 ===
// vector<NeuralInput> ReadImage400x400() {
//     vector<NeuralInput> res;
//
//     cout << "Loading 400x400 images...\n";
//
//     string catPath = "./Dataset/cat/400x400";
//     string dogPath = "./Dataset/dog/400x400";
//
//     vector<NeuralInput> cats = ReadImageFolder(catPath, 0);
//     vector<NeuralInput> dogs = ReadImageFolder(dogPath, 1);
//
//     res.insert(res.end(), cats.begin(), cats.end());
//     res.insert(res.end(), dogs.begin(), dogs.end());
//
//     cout << "Loaded " << res.size() << " images (400x400): "
//          << cats.size() << " cats, " << dogs.size() << " dogs\n";
//
//     return res;
// }
vector<NeuralInput> ReadDataset(bool isTrain, bool useSmallImage) {
    vector<NeuralInput> res;

    string basePath = "../dataset_split/";
    basePath += (isTrain ? "train/" : "test/");

    string sizeFolder = useSmallImage ? "16x16" : "400x400";

    string catPath = basePath + "Cat/" + sizeFolder;
    string dogPath = basePath + "Dog/" + sizeFolder;

    cout << "Loading "
         << (isTrain ? "TRAIN" : "TEST")
         << " dataset (" << sizeFolder << ")...\n";

    vector<NeuralInput> cats = ReadImageFolder(catPath, 0);
    vector<NeuralInput> dogs = ReadImageFolder(dogPath, 1);

    res.insert(res.end(), cats.begin(), cats.end());
    res.insert(res.end(), dogs.begin(), dogs.end());

    cout << "Loaded " << res.size() << " images: "
         << cats.size() << " cats, " << dogs.size() << " dogs\n";

    return res;
}
// void validateNetwork(NeuralNetwork<NeuralInput>& net, std::vector<NeuralInput>& testData)
// {
//     std::cout << "\n=== VALIDATION PHASE ===\n";
//     if (testData.empty()) {
//         std::cerr << "Test dataset is empty!\n";
//         return;
//     }
//     int correct = 0;
//     for (auto& sample : testData) {
//         // Bước 1: Gọi predict để xử lý
//         net.predict(&sample);
//
//         // Bước 2: Lấy kết quả từ hàm getter
//         Matrix<float>* output = (Matrix<float>*)net.getPredictResult();
//         float* outputData = outputMatrix->data();
//
//         if (!output) {
//             std::cerr << "Predict returned null for one sample.\n";
//             continue;
//         }
//
//         // Bước 3: Xử lý output
//         int predictedClass = 0;
//         float maxProb = output->get(0, 0, 0); // Giả sử get(channel, row, col)
//
//         for (int i = 1; i < 2; i++) { // 2 classes: cat(0), dog(1)
//             float prob = output->get(0, 0, i);
//             if (prob > maxProb) {
//                 maxProb = prob;
//                 predictedClass = i;
//             }
//         }
//
//         // Bước 4: So sánh với label thực
//         if (predictedClass == sample.lable) {
//             correct++;
//         }
//     }
//
//     return (float)correct / data.size();
//         // ...
// }
float elvaluate(NeuralNetwork<NeuralInput>& net, vector<NeuralInput>& data) {
    std::cout << "[DEBUG] Evaluate called, data size = " << data.size() << std::endl;
    if (data.empty()) {
        std::cerr << "[WARNING] Dataset is empty!" << std::endl;
        return 0.0f;
    }
    if (data.empty()) return 0.0f;

    int correct = 0;

    for (size_t idx = 0; idx < data.size(); idx++) {
        try {
            // Forward pass
            net.predict(&data[idx]);

            // Get output
            void* rawOutput = net.getPredictResult();
            if (!rawOutput) {
                std::cerr << "Null output at sample " << idx << "\n";
                continue;
            }

            Matrix<float>* outputMatrix = static_cast<Matrix<float>*>(rawOutput);

            // Find predicted class (argmax)
            int predictedClass = 0;
            float maxProb = -std::numeric_limits<float>::infinity();

            // Giả sử output có shape (1, 1, 2) cho 2 classes
            for (int i = 0; i < 2; i++) {
                // Dùng method get() thay vì data()
                float prob = outputMatrix->get(0, 0, i);

                if (prob > maxProb) {
                    maxProb = prob;
                    predictedClass = i;
                }
            }

            // Check correctness
            if (predictedClass == data[idx].lable) {
                correct++;
            }

        } catch (const std::exception& e) {
            std::cerr << "Error evaluating sample " << idx << ": "
                      << e.what() << "\n";
            continue;
        }
    }

    return static_cast<float>(correct) / data.size();
}


// // ============================================
// // MAIN
// // ============================================
int main() {
//     //     Matrix<Tracebackable<float>> a1 = Matrix<Tracebackable<float>>(3,16,16);
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


    vector<NeuralInput> trainingData = ReadDataset(true, useSmallImage);
    vector<NeuralInput> testData = ReadDataset(false, useSmallImage);
        //

        //
        //
        // cout << trainingData.size() << endl;
        //
        // // ============ BUILD CNN ============
        // cout << "Building CNN architecture...\n";
        //
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
        int n = 2000;

        // NeuralInput a;
        // a.lable = 1;
        // a.data = new Matrix<Tracebackable<float>>(1,1,10);
        //
        // net.predict(&a);
int epochs = 10;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float totalLoss = 0;
        for (int j = 0; j < trainingData.size(); j++) {
            net.predict(&trainingData[j]);
            net.backward();
            totalLoss += net.getError();
        }
        cout << "[Epoch " << epoch + 1 << "] Loss = " << totalLoss << endl;

        // 🔹 Đánh giá sau mỗi epoch:
        float acc = elvaluate(net, testData);
        cout << "Validation Accuracy = " << acc * 100 << "%\n";
    }
    float acc = elvaluate(net, testData);
    cout << "Test Accuracy = " << acc * 100 << "%\n";




        cout << "Architecture: Input → Conv(3→6) → Pool → FC(96→32) → FC(32→16) → Output(2)\n\n";
        //
        // // ============ TRAINING ============
    // // ============ SAVE MODEL ==========

    std::filesystem::create_directories("ReCNN/Models");

    // save the network to JSON
    ModelSerialize::saveNetwork(net, "ReCNN/Models/mymodel.json");

    std::cout << "Model saved to ReCNN/Models/mymodel.json\n";
}


