#ifndef RECNN_MODEL_SERIALIZE_H
#define RECNN_MODEL_SERIALIZE_H

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include "../Utils/json.hpp"
#include "../Component/Matrix.h"
#include "Layers/ConvolutionLayer.h"
#include "Layers/MaxPoolingLayer.h"
#include "Layers/FClayer.h"
#include "TommyDatNeuralNet/NeuralNetwork.h"

using json = nlohmann::json;
using namespace TommyDat;

class ModelSerialize {
public:

    // --- Matrix -> JSON ---
    template<typename T>
    static json MatrixToJson(const Matrix<T>& mat) {
        json j;
        dim3 d = mat.getDim();
        j["size3D"] = d.x;
        j["rows"] = d.y;
        j["cols"] = d.z;
        int len = mat.getLen();
        std::vector<T> data;
        data.reserve(len);
        for (int i = 0; i < len; ++i) {
            data.push_back(mat.getFlatten(static_cast<unsigned int>(i)));
        }
        j["data"] = std::move(data);
        return j;
    }

    // --- JSON -> Matrix ---
    template<typename T>
    static Matrix<T> loadMatrix(const json& j) {
        if (j.is_null()) throw std::runtime_error("loadMatrix: null json");

        int size3D = j.value("size3D", 1);
        int rows   = j.value("rows", 1);
        int cols   = j.value("cols", 1);
        std::vector<T> vec = j.at("data").get<std::vector<T>>();

        Matrix<T> m(size3D, rows, cols);

        for (int i = 0; i < (int)vec.size(); ++i) {
            m.setFlatten(i, vec[i]);
        }

        return m;
    }



    // --- helpers for activation enum ---
    static std::string activationToString(EnumActivationType a) {
        switch (a) {
            case EnumActivationType::ReLU: return "ReLU";
            case EnumActivationType::softMax: return "softMax";
            default: return "unknown";
        }
    }
    static EnumActivationType stringToActivation(const std::string& s) {
        if (s == "softMax") return EnumActivationType::softMax;
        return EnumActivationType::ReLU;
    }

    // --- serialize one layer (value-style JSON object) ---
    static json serializeLayer(const Layer* layer) {
        json j;
        if (auto conv = dynamic_cast<const ConvolutionLayer*>(layer)) {
            j["type"] = "ConvolutionLayer";
            j["inChannel"] = conv->getInChannel();    // if inChannel private, add getter getInChannel()
            j["outChannel"] = conv->getOutChannel();  // if private, add getter getOutChannel()
            j["kernelSize"] = conv->getKernelSize();  // or add getKernelSize()
            j["stride"] = conv->getStride();

            Matrix<float>* km = conv->getKernelMatrix(); // safe pointer (may be nullptr)
            if (km != nullptr) j["kernel"] = MatrixToJson(*km);
            else j["kernel"] = nullptr;
        }
        else if (auto pool = dynamic_cast<const MaxPoolingLayer*>(layer)) {
            j["type"] = "MaxPoolingLayer";
            j["size"] = pool->getSize();
            j["stride"] = pool->getStride();
        }
        else if (auto fc = dynamic_cast<const FClayer*>(layer)) {
            auto fc1 = (FClayer*)fc;
            j["type"] = "FClayer";
            j["dense"] = fc1->getDense();
            j["activation"] = activationToString(fc1->getActivationType());
            j["isFirst"] = fc1->isFirst();

            Matrix<float>* w = fc->getWeightMatrix();
            Matrix<float>* b = fc->getBiasMatrix();
            j["weight"] = (w ? MatrixToJson(*w) : json(nullptr));
            j["bias"]   = (b ? MatrixToJson(*b) : json(nullptr));
        }
        else {
            j["type"] = "Unknown";
        }
        return j;
    }

    // --- save whole network ---
    template<typename InputType>
    static void saveNetwork(const NeuralNetwork<InputType>& net, const std::string& filename) {
        json jNet;
        jNet["learningRate"] = net.learningRate;
        jNet["layers"] = json::array();
        for (auto l : net.layers) {
            jNet["layers"].push_back(serializeLayer(l));
        }
        // ensure parent directories exist
        std::filesystem::path p(filename);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs.is_open()) throw std::runtime_error("Cannot open file for write: " + filename);
        ofs << jNet.dump(4);
        ofs.close();
    }

    // --- deserialize one layer (create proper object safely) ---
    static Layer* deserializeLayer(const json& j) {
        std::string type = j.at("type").get<std::string>();

        if (type == "ConvolutionLayer") {
            // read essential metadata
            int inC = j.value("inChannel", 0);
            int outC = j.value("outChannel", 0);
            int ksize = j.value("kernelSize", 0);
            int stride = j.value("stride", 1);

            // construct with full constructor so kernelList is allocated
            ConvolutionLayer* conv = new ConvolutionLayer(inC, outC, ksize, stride);

            if (j.contains("kernel") && !j["kernel"].is_null()) {
                Matrix<float> km = loadMatrix<float>(j["kernel"]);
                conv->setKernelMatrix(km); // safe setter (allocates)
            }
            return conv;
        }
        else if (type == "MaxPoolingLayer") {
            int size = j.value("size", 2);
            int stride = j.value("stride", 2);
            MaxPoolingLayer* pool = new MaxPoolingLayer(stride, size);
            return pool;
        }
        else if (type == "FClayer") {
            int dense = j.value("dense", 0);
            std::string act = j.value("activation", std::string("ReLU"));
            bool isFirst = j.value("isFirst", false);

            FClayer* fc = new FClayer(dense, stringToActivation(act), isFirst);

            if (j.contains("weight") && !j["weight"].is_null()) {
                Matrix<float> w = loadMatrix<float>(j["weight"]);
                fc->setWeightMatrix(w);
            }
            if (j.contains("bias") && !j["bias"].is_null()) {
                Matrix<float> b = loadMatrix<float>(j["bias"]);
                fc->setBiasMatrix(b);
            }
            return fc;
        }

        return nullptr;
    }

    // --- load whole network ---
    template<typename InputType>
    static NeuralNetwork<InputType>* loadNetwork(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) throw std::runtime_error("Cannot open file for read: " + filename);
        json jNet;
        ifs >> jNet;
        ifs.close();

        auto net = new NeuralNetwork<InputType>();
        net->learningRate = jNet.value("learningRate", net->learningRate);

        for (auto& lj : jNet.at("layers")) {
            Layer* l = deserializeLayer(lj);
            if (l) net->add(l);
        }
        return net;
    }
};

#endif // RECNN_MODEL_SERIALIZE_H
