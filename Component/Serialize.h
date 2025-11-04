// #ifndef RECNN_MODEL_SERIALIZE_H
// #define RECNN_MODEL_SERIALIZE_H
//
// #include <fstream>
// #include <stdexcept>
// #include "../Utils/json.hpp"
// #include "../Component/Matrix.h"
// #include "Layers/ConvolutionLayer.h"
// #include "Layers/MaxPoolingLayer.h"
// #include "Layers/FClayer.h"
// #include "TommyDatNeuralNet/NeuralNetwork.h"
//
// using json = nlohmann::json;
// using namespace TommyDat;
//
// class ModelSerialize {
// public:
//     // Convert a Matrix to JSON (template kept)
//     template<typename T>
//     static json MatrixToJson(const Matrix<T>& mat) {
//         json j;
//         dim3 dim = mat.getDim();      // requires getDim() const
//         j["size3D"] = dim.x;
//         j["rows"] = dim.y;
//         j["cols"] = dim.z;
//
//         int len = mat.getLen();// getLen() must be const
//         std::vector<T> data;
//         data.reserve(len);
//         for (int i = 0; i < len; ++i) {
//             data.push_back(mat.getFlatten(static_cast<unsigned int>(i)));
//         }
//
//         j["data"] = data;
//         return j;
//     }
//
//     // Save a single layer to JSON
//     static json serializeLayer(void* layer, const std::string& type) {
//         json j;
//         j["type"] = type;
//
//         if (type == "ConvolutionLayer") {
//             ConvolutionLayer* conv = static_cast<ConvolutionLayer*>(layer);
//             j["kernel"] = MatrixToJson(conv->getWeightMatrix());
//             j["stride"] = conv->getStride();
//         } else if (type == "MaxPoolingLayer") {
//             MaxPoolingLayer* pool = static_cast<MaxPoolingLayer*>(layer);
//             j["size"] = pool->getSize();
//             j["stride"] = pool->getStride();
//         } else if (type == "FClayer") {
//             FClayer* fc = static_cast<FClayer*>(layer);
//             j["dense"] = fc->getDense();
//             // weight & bias could be nullptr
//             if (fc->getWeightMatrix() != nullptr) {
//                 j["weight"] = MatrixToJson(*fc->getWeightMatrix());
//             } else {
//                 j["weight"] = nullptr;
//             }
//
//             if (fc->getBiasMatrix() != nullptr) {
//                 j["bias"] = MatrixToJson(*fc->getBiasMatrix());
//             } else {
//                 j["bias"] = nullptr;
//             }
//         }
//
//         return j;
//     }
//
//     // Save entire NeuralNetwork
//     template<typename InputType>
//     static void saveNetwork(const NeuralNetwork<InputType>& net, const std::string& filename) {
//         json jNet;
//         jNet["learningRate"] = net.learningRate;
//         jNet["layers"] = json::array();
//
//         for (auto l : net.layers) {
//             if (auto conv = dynamic_cast<ConvolutionLayer*>(l)) {
//                 jNet["layers"].push_back(serializeLayer(conv, "ConvolutionLayer"));
//             } else if (auto pool = dynamic_cast<MaxPoolingLayer*>(l)) {
//                 jNet["layers"].push_back(serializeLayer(pool, "MaxPoolingLayer"));
//             } else if (auto fc = dynamic_cast<FClayer*>(l)) {
//                 jNet["layers"].push_back(serializeLayer(fc, "FClayer"));
//             } else {
//                 // unknown layer: optionally skip or store its type name if available
//             }
//         }
//
//         std::ofstream file(filename);
//         if (!file.is_open()) throw std::runtime_error("Failed to open file for writing: " + filename);
//         file << jNet.dump(4);
//         file.close();
//     }
//
//     // Load a matrix from JSON
//     template<typename T>
//     static Matrix<T> loadMatrix(const json& j) {
//         if (j.is_null()) throw std::runtime_error("loadMatrix: json is null");
//         int size3D = j["size3D"];
//         int rows = j["rows"];
//         int cols = j["cols"];
//         std::vector<T> data = j["data"].get<std::vector<T>>();
//         return Matrix<T>(data.data(), size3D, rows, cols);
//     }
//
//     // Load a layer from JSON
//     static void* deserializeLayer(const json& j) {
//         std::string type = j["type"];
//         if (type == "ConvolutionLayer") {
//             ConvolutionLayer* conv = new ConvolutionLayer();
//             conv->setWeightMatrix(loadMatrix<float>(j["kernel"]));
//             conv->setStride(j["stride"]);
//             return conv;
//         } else if (type == "MaxPoolingLayer") {
//             MaxPoolingLayer* pool = new MaxPoolingLayer();
//             pool->setSize(j["size"]);
//             pool->setStride(j["stride"]);
//             return pool;
//         } else if (type == "FClayer") {
//             int dense = j.value("dense", 0);
//             std::string actStr = j.value("activation", std::string("ReLU"));
//             bool isFirst = j.value("isFirst", false);
//
//             // Prefer creating with constructor that sets dense/act/isFirst if available
//             FClayer* fc = nullptr;
//             // weight & bias may be null in json (e.g., isFirst)
//             if (j.contains("weight") && !j["weight"].is_null()) {
//                 Matrix<float> w = loadMatrix<float>(j["weight"]);
//                 fc->setWeightMatrix(w);
//             }
//             if (j.contains("bias") && !j["bias"].is_null()) {
//                 Matrix<float> b = loadMatrix<float>(j["bias"]);
//                 fc->setBiasMatrix(b);
//             }
//             return fc;
//         }
//         return nullptr;
//     }
//
//     // Load entire NeuralNetwork
//     template<typename InputType>
//     static NeuralNetwork<InputType>* loadNetwork(const std::string& filename) {
//         std::ifstream file(filename);
//         if (!file.is_open()) throw std::runtime_error("Failed to open file: " + filename);
//         json jNet;
//         file >> jNet;
//         file.close();
//
//         auto net = new NeuralNetwork<InputType>();
//         net->learningRate = jNet["learningRate"];
//         for (auto& layerJson : jNet["layers"]) {
//             void* l = deserializeLayer(layerJson);
//             if (l)
//                 net->add(static_cast<Layer*>(l));
//         }
//         return net;
//     }
// };
//
// #endif //RECNN_MODEL_SERIALIZE_H
