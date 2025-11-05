//
// Created by datdau on 11/4/25.
//

#ifndef RECNN_FILE_H
#define RECNN_FILE_H
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>
using namespace std;
namespace fs = std::filesystem;
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

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>
using namespace std;
namespace fs = std::filesystem;

vector<NeuralInput> ReadImageFolder(const string& folderPath, int label, bool isTest, bool doShuffle = false) {
    vector<NeuralInput> res;

    if (!fs::exists(folderPath)) {
        cerr << "Folder not found: " << folderPath << endl;
        return res;
    }

    // Lấy tất cả đường dẫn ảnh
    vector<string> allPaths;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            string path = entry.path().string();
            if (hasImageExtension(path)) {
                allPaths.push_back(path);
            }
        }
    }

    // Sắp xếp để đảm bảo thứ tự ổn định trước khi chia train/test
    sort(allPaths.begin(), allPaths.end());

    size_t total = allPaths.size();
    if (total == 0) return res;

    size_t splitIndex = static_cast<size_t>(total * 0.7);
    size_t startIdx = isTest ? splitIndex : 0;
    size_t endIdx = isTest ? total : splitIndex;

    // Lấy phần dữ liệu tương ứng (train hoặc test)
    vector<string> selectedPaths(allPaths.begin() + startIdx, allPaths.begin() + endIdx);

    // === SHUFFLE nếu cần ===
    if (doShuffle) {
        static std::random_device rd;
        static std::mt19937 g(rd()); // bộ sinh số ngẫu nhiên
        std::shuffle(selectedPaths.begin(), selectedPaths.end(), g);
    }

    // Load ảnh
    for (const string& path : selectedPaths) {
        try {
            NeuralInput a(path);
            a.lable = label;
            res.push_back(a);
        } catch (const exception& e) {
            cerr << "Error loading " << path << ": " << e.what() << endl;
        }
    }

    return res;
}

// === HÀM ĐỌC ẢNH 16x16 ===
vector<NeuralInput> ReadImage16x16(bool isTest) {
    vector<NeuralInput> res;

    cout << "Loading 16x16 images...\n";

    string catPath = "./Dataset/cat/16x16";
    string dogPath = "./Dataset/dog/16x16";

    vector<NeuralInput> cats = ReadImageFolder(catPath, 0, isTest);
    vector<NeuralInput> dogs = ReadImageFolder(dogPath, 1, isTest);

    res.insert(res.end(), cats.begin(), cats.end());
    res.insert(res.end(), dogs.begin(), dogs.end());

    cout << "Loaded " << res.size() << " images (16x16): "
         << cats.size() << " cats, " << dogs.size() << " dogs\n";

    return res;
}


// === HÀM ĐỌC ẢNH 400x400 ===
vector<NeuralInput> ReadImage400x400(int isTest) {
    vector<NeuralInput> res;

    cout << "Loading 400x400 images...\n";

    string catPath = "./Dataset/cat/400x400";
    string dogPath = "./Dataset/dog/400x400";

    vector<NeuralInput> cats = ReadImageFolder(catPath, 0,isTest);
    vector<NeuralInput> dogs = ReadImageFolder(dogPath, 1,isTest);

    res.insert(res.end(), cats.begin(), cats.end());
    res.insert(res.end(), dogs.begin(), dogs.end());

    cout << "Loaded " << res.size() << " images (400x400): "
         << cats.size() << " cats, " << dogs.size() << " dogs\n";

    return res;
}
#endif //RECNN_FILE_H