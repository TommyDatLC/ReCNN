#include <iostream>
#include "Component/Serialize.h"
#include "Component/Matrix.h"
#include <filesystem>
#include "Component/Layers/ConvolutionLayer.h"
#include "Component/Layers/MaxPoolingLayer.h"
#include "Component/TommyDatNeuralNet/NeuralInput.h"
#include "Component/TommyDatNeuralNet/NeuralNetwork.h"


using namespace std;
using namespace TommyDat;

// make a class to backtracing
int main() {
    try {
        cout << "=== Start the model ====\n" ;

        // Create a simple neural network
        NeuralNetwork<NeuralInput> net;
        net.learningRate = 0.1;

        // Layer definitions
        // Input: RGB(3 channels)
        // Layer 1: Conv (3 -> 6 channels)
        auto conv1 = ConvolutionLayer(3, 6, 3, 1);

        // Pooling
        auto pool1 = MaxPoolingLayer(2, 2);

        // Layer 2: Conv (6 -> 12 channels)
        auto conv2 = ConvolutionLayer(6, 12, 3, 1);

        // Pooling
        auto pool2 = MaxPoolingLayer(2, 2);

        // Build model
        net.add(&conv1);
        net.add(&pool1);
        net.add(&conv2);
        net.add(&pool2);

        cout << "Network built with " << net.getSize() << "layers\n";

        // Input
        // Load one image
        NeuralInput input("./testdata1.png");
        input.lable = 1;

        // Forward pass
        cout << "Running forward pass... \n";
        net.predict(&input);

        // Prediction result
        auto output = net.getPredictResult();
        cout << "Got output matrix.\n";

        // Backward
        cout << "Running backward pass...\n";
        net.backward();

        cout << "Backward completed. Total error: " << net.CaculateError() << "\n";

        // Save the network
        std::filesystem::create_directories("Models");
        cout << "Saving network to mymodel.json...\n";
        ModelSerialize::saveNetwork(net, "Models/mymodel.json");

        // Load the network back
        cout << "Loading network from mymodel.json...\n";
        auto loadedNet = ModelSerialize::loadNetwork<NeuralInput>("Models/mymodel.json");

        // Run inference with loaded network
        cout << "Running forward pass on loaded network...\n";
        loadedNet->predict(&input);
        auto loadedOutput = loadedNet->getPredictResult();
        cout << "Loaded network output received.\n";

        cout << "=== End of ReCNN Test ===\n";
    }

    catch (const std::exception& e) {
        cerr << "Error: " << e.what() << "\n";
    }

    cout << "=== End of ReCNN Test ===\n";
    return 0;

}
