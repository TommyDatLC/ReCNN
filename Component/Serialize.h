#ifndef RECNN_MODEL_SERIALIZE_H
#define RECNN_MODEL_SERIALIZE_H

#include <fstream>
#include "../Utils/json.hpp"
#include "../Component/Matrix.h"
#include "Layers/ConvolutionLayer.h"
#include "Layers/MaxPoolingLayer.h"
#include "TommyDatNeuralNet/NeuralNetwork.h"
#include "../Component/Layers/FClayer.h"

using json = nlohmann::json;
using namespace TommyDat;

class ModelSerialize {
public:
    // === Serialize a single layer ===
    static json serializeLayer(void* layer, const std::string& type) {
        json j;
        j["type"] = type;

        if (type == "ConvolutionLayer") {
            ConvolutionLayer* conv = static_cast<ConvolutionLayer*>(layer);
            j["kernel"] = MatrixToJson(conv->getWeightMatrix());
            j["stride"] = conv->getStride();
        }
        else if (type == "MaxPoolingLayer") {
            MaxPoolingLayer* pool = static_cast<MaxPoolingLayer*>(layer);
            j["size"] = pool->getSize();
            j["stride"] = pool->getStride();
        }
        else if (type == "FClayer") {
            FClayer* fc = static_cast<FClayer*>(layer);
            j["dense"] = fc->getDense();
            j["activationType"] = static_cast<int>(fc->getActivationType());
            j["isFirst"] = fc->isFirst();

            // Serialize weights and bias
            if (fc->getWeightMatrix())
                j["weight"] = MatrixToJson(*fc->getWeightMatrix());
            if (fc->getBiasMatrix())
                j["bias"] = MatrixToJson(*fc->getBiasMatrix());
        }

        return j;
    }

    // === Convert a Matrix to JSON ===
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

    // === Save the entire network ===
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
            } else if (auto fc = dynamic_cast<FClayer*>(l)) {
                jNet["layers"].push_back(serializeLayer(fc, "FClayer"));
            }
        }

        std::ofstream file(filename);
        file << jNet.dump(4);
        file.close();
    }

    // === Load a matrix from JSON ===
    template<typename T>
    static Matrix<T> loadMatrix(const json& j) {
        int size3D = j["size3D"];
        int rows = j["rows"];
        int cols = j["cols"];
        std::vector<T> data = j["data"].get<std::vector<T>>();
        return Matrix<T>(data.data(), size3D, rows, cols);
    }

    // === Deserialize a single layer ===
    static void* deserializeLayer(const json& j) {
        std::string type = j["type"];
        if (type == "ConvolutionLayer") {
            auto* conv = new ConvolutionLayer();
            conv->setWeightMatrix(loadMatrix<float>(j["kernel"]));
            conv->setStride(j["stride"]);
            return conv;
        }
        else if (type == "MaxPoolingLayer") {
            auto* pool = new MaxPoolingLayer();
            pool->setSize(j["size"]);
            pool->setStride(j["stride"]);
            return pool;
        }
        else if (type == "FClayer") {
            int dense = j["dense"];
            auto actType = static_cast<EnumActivationType>(j["activationType"]);
            bool isFirst = j["isFirst"];

            auto* fc = new FClayer(dense, actType, isFirst);
            if (j.contains("weight"))
                fc->setWeightMatrix(loadMatrix<float>(j["weight"]));
            if (j.contains("bias"))
                fc->setBiasMatrix(loadMatrix<float>(j["bias"]));
            return fc;
        }

        return nullptr;
    }

    // === Load the entire network ===
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
                net->add(static_cast<Layer*>(l));
        }

        return net;
    }
};

#endif // RECNN_MODEL_SERIALIZE_H
