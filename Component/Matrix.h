//
// Created by datdau on 10/22/25.
//

#ifndef RECNN_TOMMYDATMATRIX_H
#define RECNN_TOMMYDATMATRIX_H
#include <stdexcept>
#include <cstring>

#include "IFlattenable.h"
#include "../Utils/GPUMatrixOp.h"
#include "../Utils/GPUmax.h"
#include "../Utils/GPUSoftMax.h"
#include "../Utils/Sum.h"

namespace TommyDat{
    template <typename T>
    class Matrix : IFlattenable<T>
    {
    public:
        Matrix(Matrix& B) {
            SetDim(B.n,B.m);
            matrixFlatten = new T[n * m];
            memcpy(matrixFlatten,B.flatten(),sizeof(n * m));

        }
        Matrix(int N,int M) {
            SetDim(N,M);
            matrixFlatten = new T[N * M];

        }
        Matrix(int N,int M,T val) {
            SetDim(N,M);
            matrixFlatten = new T[N * M];
            for (int i = 0;i < N;i++)
                for (int j = 0;j < M;j++)
                    matrixFlatten[i * M + j] = val;

        }
        Matrix(T* flattenArr,int N,int M) {
            SetDim(N,M);
            matrixFlatten = flattenArr;

        }
        Matrix(T** raw2Dmatrix,int N,int M) {
            SetDim(N,M);

            matrixFlatten = flattenArray(raw2Dmatrix);
        }
        T* flatten() override {
            return matrixFlatten;
        }


        void set(uint x,uint y,T val) {
            checkValidID(x,y);
            matrixFlatten[x * m + y] = val;

        }

        T get(uint x,uint y) {
            checkValidID(x,y);
            return matrixFlatten[x * m + y];
        }
        dim3 getDim() {
            return dim3(n,m,0);
        }

        Matrix softMax() {
            T* rawResult = new T[lenFlattenCache];
            memcpy(rawResult,matrixFlatten,sizeof(T) * lenFlattenCache);
            T maxElm = CallGPUmax(rawResult,lenFlattenCache);
            T sum = CallGPUSum(rawResult,lenFlattenCache);
            CallGPUExpMinusMax(rawResult,lenFlattenCache,maxElm);
            CallGPUSoftmax(rawResult,lenFlattenCache,sum);
            return Matrix(rawResult,n,m);
        }
        Matrix convolution(Matrix& kernel,int stride = 1) {
            if (kernel.n % 2 == 0 || kernel.m % 2 == 0) {
                throw std::runtime_error("* Cannot process kernel dimemsion % 2 != 1");
            }
            T* Bflatten = kernel.flatten();
            auto result =  CallGPUConvolution(matrixFlatten,n,m,Bflatten,kernel.n,kernel.m,stride,stride);
            return Matrix(result.newRawMatrix,result.N,result.M);
        }
        Matrix maxPooling(int size,int stride) {
            auto result =  CallGPUmaxPooling(matrixFlatten,n,m,size,stride);
            return Matrix(result.newRawMatrix,result.N,result.M);
        }
        Matrix operator*(const Matrix& B) {
            T* rawResult;
            if (m != B.n)
                throw std::runtime_error("* Dimension error, first matrix col not equal to second matrix row");
            rawResult = CallMatrixMul(matrixFlatten,B.matrixFlatten,n,m,B.m);
            return Matrix(rawResult,n,B.m);
        }
        Matrix operator+(const Matrix& B) {
            T* rawResult;
            if (m != B.m || n != B.n)
                throw std::runtime_error("+ Dimension error,the row and col must equal");
            rawResult = CallMatrixAddOrSub(matrixFlatten,B.matrixFlatten,n,m,true);
            return Matrix(rawResult,n,m);
        }
        Matrix operator-(const Matrix& B) {
            T* rawResult;
            if (m != B.m || n != B.n)
                throw std::runtime_error("- Dimension error,the row and col must equal");
            rawResult = CallMatrixAddOrSub(matrixFlatten,B.matrixFlatten,n,m,false);
            return Matrix(rawResult,n,m);
        }
        // the same with toString() function in java
        friend std::ostream& operator<<(std::ostream& os, const Matrix& a) {
            for (int i = 0; i < a.n; i++) {
                for (int j = 0; j < a.m; j++) {
                    os << a.matrixFlatten[i * a.m + j] << " ";
                }
                os << "\n";
            }
            return os;
        }
        ~Matrix() {
            // when the matrix is free;

            freeArr(matrixFlatten);

        }
        void ReLU() {
            CallGPURelu(matrixFlatten,n * m);
        }
        void heInit(int LayerSize,ull seed = 123) {
            CallGPUheInit(matrixFlatten,lenFlattenCache,LayerSize,seed);
        }
        void heInit(ull seed = 123) {
            CallGPUheInit(matrixFlatten,lenFlattenCache,lenFlattenCache,seed);
        }
        void normalize(int maxNumber = 255) {
            CallGPUnormalize(matrixFlatten,maxNumber);
        }
    private:
        T* matrixFlatten;

        int n,m;
        int lenFlattenCache = 0;
        void checkValidID(uint x,uint y) {
            if (x >= n || y >= m)
                throw std::runtime_error("out of bound");
        }
        void SetDim(int newN,int newM) {
            n = newN;
            m = newM;
            lenFlattenCache = newN * newM;
        }

    };


}
#endif //RECNN_TOMMYDATMATRIX_H