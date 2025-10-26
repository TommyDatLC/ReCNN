//
// Created by datdau on 10/25/25.
//

#ifndef RECNN_KERNEL3D_H
#define RECNN_KERNEL3D_H

#include <string>
#include <stdexcept>
#include <opencv2/opencv.hpp> // cần cài OpenCV và link khi build
#include "Matrix.h"

namespace TommyDat {

    template <typename T>
    class Kernel3D : IFlattenable<T> {
    public:
        // data sẽ chứa 'size' Matrix<T>, mỗi Matrix là một channel (rows = height, cols = width)
        // 1 mảng con trỏ matrix<T>
        Matrix<T>** data;
        Kernel3D(Matrix<T>** DATA,int SIZE,int N,int M) {
            data = DATA;
            this->size = SIZE;
            this->n = N;
            this->m = M;
        }
        Kernel3D(const Kernel3D& B) {
            size = B.size;
            m = B.m;
            n = B.n;

            Matrix<T>** copyDataFromB = new Matrix<T>*[size];
            for (int i =0 ;i < size;i++) {
                copyDataFromB[i] = new Matrix<T>(*B.data[i]);
            }
        }
        // constructor tạo kernel rỗng (size channel)
        Kernel3D(int SIZE,int N,int M,bool heInit = true) {
            this->size = SIZE;
            this->n = N;
            this->m = M;
            data = new Matrix<T>*[size];
            for (int i =0 ;i < size;i++) {
                data[i] = new Matrix<T>(N,M);
                if (heInit)
                    data[i]->heInit();
            }

        }


        // Constructor: đọc ảnh từ path và đưa từng channel vào data[]
        Kernel3D(const std::string& path) {
            // đọc ảnh với OpenCV, giữ nguyên số kênh (IMREAD_UNCHANGED)
            cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
            if (img.empty()) {
                throw std::runtime_error(std::string("Cannot read image from: ") + path);
            }

            int height = img.rows;
            int width  = img.cols;
            int channels = img.channels();

            if (channels <= 0) {
                throw std::runtime_error("Image has no channels");
            }

            // lưu kích thước
            this->size = channels;
            this->n = height;
            this->m = width;

            // cấp phát mảng con trỏ cho data
            data = new Matrix<T>*[channels];

            // chuẩn hoá ảnh về 8-bit nếu cần
            cv::Mat img8;
            if (img.depth() != CV_8U) {
                img.convertTo(img8, CV_8U);
            } else {
                img8 = img;
            }

            // Nếu muốn tách nhanh: dùng cv::split (tránh vòng for pixel)
            std::vector<cv::Mat> splits;
            cv::split(img8, splits); // splits.size() == channels

            for (int ch = 0; ch < channels; ++ch) {
                // Mỗi splits[ch] là single-channel CV_8U với kích thước height x width
                // Cấp phát buffer flat kiểu T
                T* flat = new T[height * width];

                // copy dữ liệu từ cv::Mat (uchar) sang flat (T)
                for (int r = 0; r < height; ++r) {
                    const uchar* rowPtr = splits[ch].ptr<uchar>(r);
                    for (int c = 0; c < width; ++c) {
                        uchar val = rowPtr[c]; // vì split rồi nên mỗi pixel là 1 byte
                        flat[r * width + c] = static_cast<T>(val);
                    }
                }

                // Tạo Matrix từ flat (Matrix sẽ nhận ownership nếu impl của bạn đúng)
                data[ch] = new Matrix<T>(flat, height, width);
            }
        }
        // trả về số channel
        dim3 getDim() const {
            return dim3(size,n,m);
        }

        ~Kernel3D() {
            // xóa mảng Matrix (Matrix destructor sẽ giải phóng memory bên trong)
            if (data) {
                for (int i =0 ;i < size;i++) {
                    delete data[i];
                }
            }
            delete[] data;
        }
        T* flatten() override {
            if (data == nullptr || size <= 0)
                throw std::runtime_error("Kernel3D is empty or uninitialized");

            // Giả định tất cả channel có cùng kích thước
            dim3 dim = data[0]->getDim();
            int rows = dim.x;
            int cols = dim.y;
            int channelSize = rows * cols;

            // Tổng số phần tử = size * rows * cols
            T* flat = new T[size * channelSize];

            // Duyệt qua từng channel và copy dữ liệu
            for (int ch = 0; ch < size; ++ch) {
                T* channelFlat = data[ch]->flatten();
                std::memcpy(flat + ch * channelSize, channelFlat, sizeof(T) * channelSize);
            }

            return flat;
        }
        void heInit() {
            for (int i = 0; i < size; ++i)
                data[i]->heInit();
        }

        friend std::ostream& operator<<(std::ostream& os,const Kernel3D& B) {
            for (int i = 0;i < B.size;i++)
                os << "matrix:" << i << '\n' << *B.data[i] << '\n';
            return os;
        }
    private:
        int n,m;
        int size = 0;
    };

} // namespace TommyDat

#endif // RECNN_KERNEL3D_H
