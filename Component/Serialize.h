#ifndef RECNN_MODEL_SERIALIZE_H
#define RECNN_MODEL_SERIALIZE_H

#include <fstream>
#include "Utils/json.hpp"
#include "Component/Matrix.h"
#include "Layers/ConvolutionLayer.h"
#include "Layers/MaxPoolingLayer.h"
#include "TommyDatNeuralNet/NeuralNetwork.h"

using json = nlohmann::json;
using namespace TommyDat;

class ModelSerialize {
public:
    // Save a single layer to JSON
    static json serializeLayer(void* layer, const std::string& type) {
        json j;
        j["type"] = type;

        if (type == "ConvolutionLayer") {
            ConvolutionLayer* conv = static_cast<ConvolutionLayer*>(layer);
            // save kernel weights
            j["kernel"] = MatrixToJson(conv->getWeightMatrix());
            j["stride"] = conv->stride;
        } else if (type == "MaxPoolingLayer") {
            MaxPoolingLayer* pool = static_cast<MaxPoolingLayer*>(layer);
            j["size"] = pool->size;
            j["stride"] = pool->stride;
        }
        // add more layer types here if needed

        return j;
    }

    // Convert a Matrix to JSON
    template<typename T>
    static json MatrixToJson(const Matrix<T>& mat) {
        json j;
        dim3 dim = mat.getDim();
        j["size3D"] = dim.x;
        j["rows"] = dim.y;
        j["cols"] = dim.z;
        std::vector<T> data(mat.getLen());
        memcpy(data.data(), mat.flatten(), sizeof(T) * mat.getLen());
        j["data"] = data;
        return j;
    }

    // Save entire NeuralNetwork
    template<typename InputType>
    static void saveNetwork(const NeuralNetwork<InputType>& net, const std::string& filename) {
        json jNet;
        jNet["learningRate"] = net.learningRate;
        jNet["layers"] = json::array();

        for (auto l : net.layers) {
            if (auto conv = dynamic_cast<ConvolutionLayer*>(l)) {
                jNet["layers"].push_back(serializeLayer(conv, "ConvolutionLayer"));
            } else if (auto pool = dynamic_cast<MaxPoolingLayer*>(l)) {
                jNet["layers"].push_back(serializeLayer(pool, "MaxPoolingLayer"));
            }
        }

        std::ofstream file(filename);
        file << jNet.dump(4);
        file.close();
    }

    // Load a matrix from JSON
    template<typename T>
    static Matrix<T> loadMatrix(const json& j) {
        int size3D = j["size3D"];
        int rows = j["rows"];
        int cols = j["cols"];
        std::vector<T> data = j["data"].get<std::vector<T>>();
        return Matrix<T>(data.data(), size3D, rows, cols);
    }

    // Load a layer from JSON
    static void* deserializeLayer(const json& j) {
        std::string type = j["type"];
        if (type == "ConvolutionLayer") {
            ConvolutionLayer* conv = new ConvolutionLayer();
            conv->setWeightMatrix(loadMatrix<float>(j["kernel"])); // adjust T if needed
            conv->stride = j["stride"];
            return conv;
        } else if (type == "MaxPoolingLayer") {
            MaxPoolingLayer* pool = new MaxPoolingLayer();
            pool->size = j["size"];
            pool->stride = j["stride"];
            return pool;
        }
        return nullptr;
    }

    // Load entire NeuralNetwork
    template<typename InputType>
    static NeuralNetwork<InputType>* loadNetwork(const std::string& filename) {
        std::ifstream file(filename);
        json jNet;
        file >> jNet;
        file.close();

        auto net = new NeuralNetwork<InputType>();
        net->learningRate = jNet["learningRate"];
        for (auto& layerJson : jNet["layers"]) {
            void* l = deserializeLayer(layerJson);
            if (l)
                net->add(static_cast<typename NeuralNetwork<InputType>::LayerBase*>(l));
        }
        return net;
    }
};

#endif //RECNN_MODEL_SERIALIZE_H
