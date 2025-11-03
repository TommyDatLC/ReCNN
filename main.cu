#include <iostream>
#include "Component/TommyDatNeuralNet/NeuralInput.h"
#include "Component/TommyDatNeuralNet/NeuralNetwork.h"
#include "Component/Matrix.h"
#include "Component/Layers/ConvolutionLayer.h"
#include "Component/Layers/MaxPoolingLayer.h"
#include "Component/Layers/FClayer.h"


using namespace std;
using namespace TommyDat;

NeuralInput loadData(bool useSmall, int label) {
    string path;
    if (useSmall) {
        path = "./dog_16x16.JPEG";
        cout << "Loading 16x16: " << path << endl;
    }
    else {
        path = "./dog_400x400.jpg";
        cout << "Loading 400x400: " << path << endl;
    }
    NeuralInput input(path);
    input.lable = label;  // Set label (0 = Dog, 1 = Cat)
    cout << "Loaded image successfully: " << path << endl;
    return input;
}

int main () {

    try {


        // ============ LOAD IMAGE ============
        bool useSmallImage = true;
        int trueLabel = 0;  // Giả sử ảnh này là Dog (0), nếu Cat thì = 1
        NeuralInput input = loadData(useSmallImage, trueLabel);

        // ============ BUILD CNN ============
        cout << "\nBuilding CNN architecture...\n";

        // Convolutional layers
        auto layer1 = ConvolutionLayer(3, 6, 3, 2);   // 3→6 channels, kernel 3×3, stride 2
        auto layer2 = MaxPoolingLayer(2, 2);          // 2×2 pooling

        // Fully connected layers (MLP)
        // Giả sử sau conv+pool: 6×4×4 = 96 neurons (tùy input size)
        auto fc1 = FClayer(96, 32, EnumActivationType::ReLU);    // 96 → 32
        auto fc2 = FClayer(32, 16, EnumActivationType::ReLU);    // 32 → 16
        auto output = FClayer(16, 2, EnumActivationType::softMax);  // 16 → 2 (Dog, Cat) - Will apply softMax in forward

        // Chain layers
        layer1.setNextLayer(&layer2);
        layer2.setNextLayer(&fc1);
        fc1.setNextLayer(&fc2);
        fc2.setNextLayer(&output);

        fc1.setLastLayer(&layer2);
        fc2.setLastLayer(&fc1);
        output.setLastLayer(&fc2);

        cout << "Architecture:\n";
        cout << "  Input → Conv(3→6) → Pool → FC(96→32) → FC(32→16) → Output(2)\n\n";

        // ============ FORWARD PASS ============
        cout << "Running forward pass...\n";
        layer1.inference(&input);

        // ============ GET PREDICTION ============

        // Lấy output từ last FC layer (đã có softmax!)
        Matrix<Tracebackable<float>>* rawOutput = output.getOutActivation();

        if (!rawOutput) {
            cerr << " Error: No output from network!" << endl;
            return 1;
        }

        // Convert to float values (already softmax probabilities)
        Matrix<float> probabilities = toValueMatrix<float>(*rawOutput);

        const char* classes[] = {"Dog ", "Cat "};



        // Get probabilities
        float prob_dog = probabilities.getFlatten(0);
        float prob_cat = probabilities.getFlatten(1);

        cout << classes[0] << ": " << (prob_dog * 100.0f) << "%\n";
        cout << classes[1] << ": " << (prob_cat * 100.0f) << "%\n";

        // Find prediction (argmax)
        int prediction;
        if (prob_dog > prob_cat) {
            prediction = 0;
        } else {
            prediction = 1;
        }


        cout << "\n>>> PREDICTION: " << classes[prediction] << "\n";

        // Calculate error/loss (Cross-Entropy)
        float loss = -logf(probabilities.getFlatten(trueLabel));
        cout << ">>> LOSS (Cross-Entropy): " << loss << "\n";



        // ============ TRAINING  ============
        bool doTrain = false;  // Set true to train

        if (doTrain) {
            cout << "\n=== TRAINING MODE ===\n\n";


            // gradient = softmax - target (one-hot)

            Matrix<float> gradient(probabilities);  // Copy probabilities

            // Subtract 1 at true label position
            gradient.setFlatten(trueLabel, gradient.getFlatten(trueLabel) - 1.0f);

            // Backward pass
            float learningRate = 0.01f;
            output.backward(&gradient, learningRate);

            cout << "Training step completed!\n";
            cout << "Weights updated with learning rate: " << learningRate << "\n";
        }

        cout << "\nProgram finished successfully!\n";

    }
    catch (const exception& e) {
        cerr << "\n Error: " << e.what() << endl;
        return 1;
    }

    return 0;

}