#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

#include "Component/Layers/ConvolutionLayer.h"
#include "Component/Layers/FClayer.h"
#include "Component/Layers/MaxPoolingLayer.h"
#include "Component/TommyDatNeuralNet/NeuralInput.h"


#include "Component/TommyDatNeuralNet/NeuralNetwork.h"

namespace TommyDat {
    class FCInput;
}

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

    //     Matrix a = Matrix<float>(1,1,16);
    // cout << a;
        // cout << "========================================\n";
        // cout << "   Cat vs Dog CNN Classifier\n";
        // cout << "========================================\n\n";
        NeuralNetwork<NeuralInput> net;
        //
        // // ============ LOAD IMAGES ============
        // bool useSmallImage = true;  // true = 16x16, false = 400x400
        // vector<NeuralInput> trainingData;
        //
        // if (useSmallImage) {
        //     trainingData = ReadImage16x16();
        // } else {
        //     trainingData = ReadImage400x400();
        // }
        //
        //
        // cout << trainingData.size() << endl;
        //
        // // ============ BUILD CNN ============
        // cout << "Building CNN architecture...\n";
        //
        // //auto layer1 = ConvolutionLayer(3, 6, 3, 2);
        // //auto layer2 = MaxPoolingLayer(2, 2);
        auto fc1 = FClayer(10, EnumActivationType::ReLU,true);
        auto fc2 = FClayer( 16, EnumActivationType::ReLU);
        auto output = FClayer(2, EnumActivationType::softMax);
        net.add(&fc1);
        net.add(&fc2);
        net.add(&output);
        fc1.init();
        fc2.init();
        output.init();
        NeuralInput a;
        a.lable = 1;
         a.data = new Matrix<Tracebackable<float>>(1,1,10);
        net.predict(&a);
        auto t = output.getOutActivation();
        std::cout << "OutputMatrix" <<*t ;
        //
        //
        // cout << "Architecture: Input → Conv(3→6) → Pool → FC(96→32) → FC(32→16) → Output(2)\n\n";
        //
        // // ============ TRAINING ============


}