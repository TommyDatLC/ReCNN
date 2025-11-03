#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include "Component/Serialize.h"
#include "Component/Matrix.h"
#include "Component/Layers/ConvolutionLayer.h"
#include "Component/Layers/MaxPoolingLayer.h"
#include "Component/TommyDatNeuralNet/NeuralInput.h"
#include "Component/TommyDatNeuralNet/NeuralNetwork.h"

using namespace std;
using namespace TommyDat;
namespace fs = std::filesystem;

// minimal dataset loader
vector<NeuralInput> loadDataset(const string& datasetRoot) {
vector<NeuralInput> dataset;

        int labelId = 0;

        // iterate over class folders (e.g., "cat", "dog")
        for (auto& classDir : fs::directory_iterator(datasetRoot)) {
            if (!classDir.is_directory()) continue;

            // iterate subfolders in class folder and pick only "16x16-*" folders
            for (auto& subDir : fs::directory_iterator(classDir.path())) {
                if (!subDir.is_directory()) continue;

                string subName = subDir.path().filename().string();
                if (subName.size() >= 6 && subName.compare(0, 6, "16x16-") == 0) {
                    // iterate files in the 16x16-* folder
                    for (auto& imgFile : fs::directory_iterator(subDir.path())) {
                        if (!imgFile.is_regular_file()) continue;

                        string ext = imgFile.path().extension().string();
                        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") continue;

                        NeuralInput input(imgFile.path().string());
                        input.lable = labelId;  // assign numeric label
                        dataset.push_back(input);
                    }
                }
            }

            labelId++;  // increment label per class
        }

        return dataset;

        }


int main() {
try {
cout << "=== Start the model ====\n";

    // Create a simple neural network
    NeuralNetwork<NeuralInput> net;
    net.learningRate = 0.1;

    auto conv1 = ConvolutionLayer(3, 6, 3, 1);
    auto pool1 = MaxPoolingLayer(2, 2);
    auto conv2 = ConvolutionLayer(6, 12, 3, 1);
    auto pool2 = MaxPoolingLayer(2, 2);

    net.add(&conv1);
    net.add(&pool1);
    net.add(&conv2);
    net.add(&pool2);

    cout << "Network built with " << net.getSize() << " layers\n";

    // Load dataset
    string datasetPath = "../Dataset";  // adjust path
    auto dataset = loadDataset(datasetPath);
    cout << "Loaded " << dataset.size() << " images.\n";

    // Run forward/backward for each sample
    for (auto& sample : dataset) {
        net.predict(&sample);
        auto output = net.getPredictResult();
        net.backward();
        cout << "Processed sample with label=" << sample.lable
             << " , error=" << net.CaculateError() << "\n";
    }

    // Save network
    fs::create_directories("Models");
    ModelSerialize::saveNetwork(net, "Models/mymodel.json");
    cout << "Network saved.\n";

    // Load network
    auto loadedNet = ModelSerialize::loadNetwork<NeuralInput>("Models/mymodel.json");
    cout << "Network loaded.\n";

    // Run inference with loaded network
    for (auto& sample : dataset) {
        loadedNet->predict(&sample);
        auto loadedOutput = loadedNet->getPredictResult();
    }

    cout << "=== End of ReCNN Test ===\n";
}
catch (const exception& e) {
    cerr << "Error: " << e.what() << "\n";
}

return 0;

}
