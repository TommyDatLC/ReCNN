//
// Created by datdau on 10/22/25.
//

#ifndef RECNN_TOMMYDATMATRIX_H
#define RECNN_TOMMYDATMATRIX_H
#include <stdexcept>
#include <cstring>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>   // <-- thêm include OpenCV

#include "../Utils/GPUMatrixOp.h"
#include "../Utils/GPUmax.h"
#include "../Utils/GPUSoftMax.h"
#include "../Utils/Sum.h"

namespace TommyDat{
    template <typename T>
    class Matrix
    {
    public:
        // copy constructor (sửa memcpy)
        Matrix(Matrix& B) {
            SetDim(B.size3D,B.n,B.m);
            matrixFlatten = new T[lenFlattenCache];
            memcpy(matrixFlatten, B.flatten(), sizeof(T) * lenFlattenCache);
        }

        Matrix(int size3D,int N,int M) {
            SetDim(size3D,N,M);
            matrixFlatten = new T[lenFlattenCache];
            heInit();
        }
        Matrix(int size3D,int N,int M,T val) {
            SetDim(size3D,N,M);
            matrixFlatten = new T[lenFlattenCache];
            for (int s = 0;s < size3D;s++)
                for (int i = 0;i < N;i++)
                    for (int j = 0;j < M;j++)
                        matrixFlatten[s * N * M + i * M + j] = val;
        }
        Matrix(T* flattenArr,int size3D,int N,int M) {
            SetDim(size3D,N,M);
            matrixFlatten = flattenArr;
        }
        Matrix(T*** raw3Dmatrix,int size3D,int N,int M) {
            SetDim(size3D,N,M);
            matrixFlatten = flattenArray(raw3Dmatrix,size3D,N,M);
        }

        // --- NEW: constructor đọc ảnh từ path (sử dụng OpenCV) ---
        // Đọc ảnh giữ nguyên số channel (IMREAD_UNCHANGED), chuẩn hoá về 8-bit và map mỗi channel vào 1 "slice".
        Matrix(const std::string& path) {
            cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
            if (img.empty()) {
                throw std::runtime_error(std::string("Cannot read image from: ") + path);
            }

            int height = img.rows;
            int width  = img.cols;
            int channels = 3;

            if (channels <= 0) {
                throw std::runtime_error("Image has no channels");
            }

            // Set kích thước (size3D = channels)
            SetDim(channels, height, width);

            // Chuẩn hoá về CV_8U nếu cần
            cv::Mat img8;
            if (img.depth() != CV_8U) {
                // convertTo sẽ scale/clamp nếu cần, nhưng đây là cách nhanh gọn.
                img.convertTo(img8, CV_8U);
            } else {
                img8 = img;
            }

            // split thành channels (each is single-channel CV_8U)
            std::vector<cv::Mat> splits;
            cv::split(img8, splits); // splits.size() == channels

            // allocate flatten buffer
            matrixFlatten = new T[lenFlattenCache];

            for (int ch = 0; ch < channels; ++ch) {
                // đảm bảo kích thước hợp lệ
                if (splits[ch].rows != height || splits[ch].cols != width) {
                    // unexpected but safe-check
                    throw std::runtime_error("Unexpected split channel size");
                }

                for (int r = 0; r < height; ++r) {
                    const uchar* rowPtr = splits[ch].ptr<uchar>(r);
                    for (int c = 0; c < width; ++c) {
                        uchar val = rowPtr[c];
                        // index layout: channel-major, then row-major within channel
                        matrixFlatten[ch * (height * width) + r * width + c] = val;
                    }
                }
            }
        }
        // --- END new constructor ---

        T* flatten() {
            return matrixFlatten;
        }

        void set(uint id3d,uint x,uint y,T val) {
            checkValidID(x,y);
            matrixFlatten[id3d * m * n + x * m + y] = val;
        }
        void setFlatten(uint id,T val) {
            matrixFlatten[id] = val;
        }

        T get(uint id3d,uint x,uint y) {
            checkValidID(x,y);
            return matrixFlatten[id3d * m * n + x * m + y];
        }
        T getFlatten(uint id) {
            return matrixFlatten[id];
        }
        dim3 getDim() {
            return dim3(size3D,n,m);
        }
        int getLen() {
            return lenFlattenCache;
        }
        Matrix* softMax() {
            T* rawResult = new T[lenFlattenCache];
            memcpy(rawResult,matrixFlatten,sizeof(T) * lenFlattenCache);
            T maxElm = CallGPUmax(rawResult,lenFlattenCache);
            T sum = CallGPUSum(rawResult,lenFlattenCache);
            CallGPUExpMinusMax(rawResult,lenFlattenCache,maxElm);
            CallGPUSoftmax(rawResult,lenFlattenCache,sum);
            return new Matrix(rawResult,size3D,n,m);
        }
        Matrix* convolution(Matrix& kernel,int stride = 1) {
            if (kernel.n % 2 == 0 || kernel.m % 2 == 0) {
                throw std::runtime_error("* Cannot process kernel dimemsion % 2 != 1");
            }
            T* kernelFlatten = kernel.flatten();
            auto result =  CallGPUConvolution(matrixFlatten,size3D,n,m,kernelFlatten,kernel.size3D,kernel.n,kernel.m,stride);
            return new Matrix(result.newRawMatrix,result.Size3D,result.N,result.M);
        }
        Matrix* maxPooling(int size,int stride) {
            auto result =  CallGPUmaxPooling(matrixFlatten,size3D,n,m,size,stride);
            return new Matrix(result.newRawMatrix,result.Size3D,result.N,result.M);
        }
        Matrix* operator*(const Matrix& B) {
            T* rawResult;
            if (m != B.n)
                throw std::runtime_error("* Dimension error, first matrix col not equal to second matrix row");
            if (size3D != 1)
                throw std::runtime_error(" We haven't support 3D matrix mul yet");
            rawResult = CallMatrixMul(matrixFlatten,B.matrixFlatten,n,m,B.m);
            return new Matrix(rawResult,n,B.m);
        }
        Matrix* operator+(const Matrix& B) {
            T* rawResult;
            if (m != B.m || n != B.n || size3D != B.size3D)
                throw std::runtime_error("+ Dimension error,the row and col must equal");
            rawResult = CallMatrixAddOrSub(matrixFlatten,B.matrixFlatten,lenFlattenCache,true);
            return new Matrix(rawResult,size3D,n,m);
        }
        Matrix* operator-(const Matrix& B) {
            T* rawResult;
            if (m != B.m || n != B.n || size3D != B.size3D)
                throw std::runtime_error("- Dimension error,the row and col must equal");
            rawResult = CallMatrixAddOrSub(matrixFlatten,B.matrixFlatten,lenFlattenCache,false);
            return new Matrix(rawResult,size3D,n,m);
        }
        // the same with toString() function in java
        friend std::ostream& operator<<(std::ostream& os, Matrix& a) {
            os << " size:" << a.size3D << 'x' << a.n << 'x' << a.m << '\n';
            for (int s = 0 ;s < a.size3D;s++) {
                os << "matrix:" << s << '\n';
                for (int i = 0; i < a.n; i++) {
                    for (int j = 0; j < a.m; j++) {
                        os << a.get(s,i,j) << " ";
                    }
                    os << "\n";
                }
            }
            return os;
        }


        ~Matrix() {
            // when the matrix is free;
            freeArr(matrixFlatten);
        }
        void ReLU() {
            CallGPURelu(matrixFlatten,lenFlattenCache);
        }
        void heInit(int LayerSize,ull seed = 0) {
            CallGPUheInit(matrixFlatten,lenFlattenCache,LayerSize,seed);
        }
        void heInit(ull seed = 0) {
            CallGPUheInit(matrixFlatten,lenFlattenCache,lenFlattenCache,seed);
        }

        void normalize(T maxN = 255) {
            CallGPUNormalize(matrixFlatten,lenFlattenCache,maxN);
        }
        template <typename TDeviceFunction>
        void apply(TDeviceFunction deviceFuntion) {
            CallGPUapply(matrixFlatten,lenFlattenCache,deviceFuntion);
        }
        void log() {
            apply([] __device__ (T x) {
                if (x != 0)
                    return __logf(x.get());
                else
                    return 0.f;
            });
        }
        void transpose() {
            CallGPUTranspose(matrixFlatten,size3D,n,m);
        }
        void reShape(int newSize3D,int newM,int newN) {
            if (newSize3D * newN * newM != lenFlattenCache)
                throw std::runtime_error("cannot reshape because the new size not match the old size");
            SetDim(newSize3D,newM,newN);
        }

        static Matrix* mulUnofficial(Matrix& a,Matrix& b) {
            dim3 aDim = a.getDim(),
                bDim = b.getDim();
            int aLen = aDim.x * aDim.y * aDim.z ;
            if (aLen!= bDim.x * bDim.y * bDim.z)
                throw std::runtime_error("Dimension error, cannot mul when the two matrix shape is not match");
            auto ptr_res = CallGPUmatrixBasicOP(a.flatten(),b.flatten(),aLen,MAT_OP_MUL);
            return new Matrix(ptr_res,aDim.x,aDim.y,aDim.z);
        }
    private:
        T* matrixFlatten = nullptr;
        int n=0,m=0,size3D=0;
        int lenFlattenCache = 0;
        void checkValidID(uint x,uint y) {
            if (x >= n || y >= m)
                throw std::runtime_error("out of bound");
        }
        void SetDim(int newSize3D,int newN,int newM) {
            size3D = newSize3D;
            n = newN;
            m = newM;
            lenFlattenCache = newN * newM * size3D;
        }

    };


}
#endif //RECNN_TOMMYDATMATRIX_H
