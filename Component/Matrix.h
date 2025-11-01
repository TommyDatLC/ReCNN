//
// Created by datdau on 10/22/25.
//

#ifndef RECNN_TOMMYDATMATRIX_H
#define RECNN_TOMMYDATMATRIX_H
#include <stdexcept>
#include <cstring>
#include <string>
#define STB_IMAGE_IMPLEMENTATION

#include "../Utils/stbImage.h"
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
            Matrix(dim3 t) {
                SetDim(t);
                matrixFlatten = new T[lenFlattenCache];
                heInit();
            }
            Matrix(int size3D,int N,int M,T val) {
                InitMatrixWithVal(size3D,N,M,val);
            }
            Matrix(dim3 t,T val) {
                InitMatrixWithVal(t.x,t.y,t.z,val);
            }
            Matrix(T* flattenArr,int size3D,int N,int M) {
                SetDim(size3D,N,M);
                matrixFlatten = flattenArr;
            }
            Matrix(T* flattenArr,dim3 size) {
                SetDim(size);
                matrixFlatten = flattenArr;
            }
            Matrix(T*** raw3Dmatrix,int size3D,int N,int M) {
                SetDim(size3D,N,M);
                matrixFlatten = flattenArray(raw3Dmatrix,size3D,N,M);
            }

            // --- NEW: constructor đọc ảnh từ path (sử dụng OpenCV) ---
            // Đọc ảnh giữ nguyên số channel (IMREAD_UNCHANGED), chuẩn hoá về 8-bit và map mỗi channel vào 1 "slice".
    // đảm bảo file này có trong include path

    // ... trong class Matrix<T> ...
            Matrix(const std::string& path)
            {
                int width = 0, height = 0, channels_in_file = 0;
                const int desired_channels = 3; // ta chuẩn hoá về 3 channels (RGB)

                // stbi_load trả về pointer tới dữ liệu unsigned char (row-major, top-left)
                // Pixel layout: R G B R G B ...
                unsigned char* imgData = stbi_load(path.c_str(), &width, &height, &channels_in_file, desired_channels);
                if (!imgData) {
                    throw std::runtime_error(std::string("Cannot read image from: ") + path + " : " + stbi_failure_reason());
                }

                // Nếu file có chiều sâu khác, stb đã convert cho ta sang 8-bit per channel khi trả về imgData.
                int channels = desired_channels;
                if (channels <= 0) {
                    stbi_image_free(imgData);
                    throw std::runtime_error("Image has no channels");
                }

                // Set kích thước (size3D = channels)
                SetDim(channels, height, width);

                // allocate flatten buffer (giữ cách index channel-major như trước)
                // mình giả sử lenFlattenCache đã được tính trong SetDim
                matrixFlatten = new T[lenFlattenCache];

                // stbi trả dữ liệu theo row-major per-channel interleaved (RGBRGB...), ta cần chuyển sang layout channel-major
                // index layout mong muốn: channel-major, then row-major within channel
                // dst_index = ch * (height*width) + r * width + c
                const int hw = height * width;
                for (int r = 0; r < height; ++r) {
                    for (int c = 0; c < width; ++c) {
                        int base = (r * width + c) * channels; // offset trong imgData
                        for (int ch = 0; ch < channels; ++ch) {
                            unsigned char val = imgData[base + ch];
                            // Khởi tạo T từ giá trị byte (0..255).
                            // Dùng T(val) để hỗ trợ cả T = float và T = Tracebackable<float> (explicit ctor OK).
                            matrixFlatten[ch * hw + r * width + c] = T(val);
                        }
                    }
                }

                // Giải phóng bộ nhớ của stb
                stbi_image_free(imgData);
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
                if (id >= lenFlattenCache)
                    throw std::runtime_error("Out of bound");
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
                CallGPUExpMinusMax(rawResult,lenFlattenCache,maxElm);
                T sum = CallGPUSum(rawResult,lenFlattenCache);
                CallGPUSoftmax(rawResult,lenFlattenCache,sum);
                return new Matrix(rawResult,size3D,n,m);
            }
            template <typename Tker>
            Matrix* convolution(Matrix<Tker>& kernel,int stride = 1) {
                dim3 dimKer = kernel.getDim();
                if (dimKer.y % 2 == 0 || dimKer.z % 2 == 0) {
                    throw std::runtime_error("* Cannot process kernel dimemsion % 2 != 1");
                }
                Tker* kernelFlatten = kernel.flatten();
                auto result =  CallGPUConvolution(matrixFlatten,size3D,n,m,kernelFlatten,dimKer.x,dimKer.y,dimKer.z,stride);
                return new Matrix(result.newRawMatrix,result.Size3D,result.N,result.M);
            }
            Matrix* maxPooling(int size,int stride) {
                auto result =  CallGPUmaxPooling(matrixFlatten,size3D,n,m,size,stride);
                return new Matrix(result.newRawMatrix,result.Size3D,result.N,result.M);
            }
            template <typename T2,typename TOut = T>
            Matrix<TOut>* operator*(const Matrix<T2>& B) {
                T* rawResult;
                if (m != B.n)
                    throw std::runtime_error("* Dimension error, first matrix col not equal to second matrix row");
                if (size3D != 1)
                    throw std::runtime_error(" We haven't support 3D matrix mul yet");
                rawResult = CallMatrixMul(matrixFlatten,B.matrixFlatten,n,m,B.m);
                return new Matrix<TOut>(rawResult,n,B.m);
            }
            template <typename T2>
            Matrix* operator+(Matrix<T2>& B) {
                T* rawResult;
                checkValidBasicOp(B);
                rawResult = CallGPUmatrixBasicOP(matrixFlatten,B.flatten(),lenFlattenCache,true);
                return new Matrix<T>(rawResult,size3D,n,m);
            }
            template <typename T2>
            Matrix<T>* operator-(Matrix<T2>& B) {
                T* rawResult;
                checkValidBasicOp<T2>(B);
                rawResult = CallGPUmatrixBasicOP(matrixFlatten,B.flatten(),lenFlattenCache,false);
                return new Matrix<T>(rawResult,size3D,n,m);
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




        private:
            T* matrixFlatten = nullptr;
            int n=0,m=0,size3D=0;
            int lenFlattenCache = 0;
            void checkValidID(uint x,uint y) {
                if (x >= n || y >= m)
                    throw std::runtime_error("out of bound");
            }
            template <typename T2>
            void checkValidBasicOp(Matrix<T2>& B) {
                dim3 dimB = B.getDim();
                if (n != dimB.y || m != dimB.z || size3D != dimB.x)
                    throw std::runtime_error("Dimension error,the row and col must equal");
            }
            void SetDim(int newSize3D,int newN,int newM) {
                size3D = newSize3D;
                n = newN;
                m = newM;
                lenFlattenCache = newN * newM * size3D;
            }
            void SetDim(dim3 t) {
                SetDim(t.x,t.y,t.z);
            }
            void InitMatrixWithVal(int size3D,int N,int M,T val) {
                SetDim(size3D,N,M);
                matrixFlatten = new T[lenFlattenCache];
                for (int s = 0;s < size3D;s++)
                    for (int i = 0;i < N;i++)
                        for (int j = 0;j < M;j++)
                            matrixFlatten[s * N * M + i * M + j] = val;
            }

        };
    template<typename TOut>
    Matrix<TOut> toValueMatrix(Matrix<Tracebackable<TOut>>& src) {
        dim3 dim = src.getDim();  // (size3D, n, m)
        Matrix<TOut> dst(dim.x, dim.y, dim.z);
        int total = dim.x * dim.y * dim.z;
        for (int i = 0; i < total; i++) {
            dst.flatten()[i] = src.flatten()[i].get();
        }
        return dst;
    }
    template <typename T1,typename T2>
    static Matrix<T1>* mulUnofficial(Matrix<T1>& a,Matrix<T2>& b) {
        dim3 aDim = a.getDim(),
            bDim = b.getDim();
        int aLen = aDim.x * aDim.y * aDim.z ;
        if (aLen!= bDim.x * bDim.y * bDim.z)
            throw std::runtime_error("Dimension error, cannot mul when the two matrix shape is not match");
        auto ptr_res = CallGPUmatrixBasicOP(a.flatten(),b.flatten(),aLen,MAT_OP_MUL);
        return new Matrix<T1>(ptr_res,aDim.x,aDim.y,aDim.z);
    }
}


#endif //RECNN_TOMMYDATMATRIX_H
