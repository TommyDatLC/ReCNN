//
// Created by datdau on 10/22/25.
//

#ifndef RECNN_TOMMYDATMATRIX_H
#define RECNN_TOMMYDATMATRIX_H
#include <stdexcept>
#include <version>
#include <cstring>
#include "../Utils/GPUMatrixOp.h"
#include "../Utils/GPUmax.h"
#include "../Utils/GPUSoftMax.h"

namespace TommyDat{
    template <typename T>
    class Matrix {
    public:
        Matrix(int N,int M,T val) {
            SetDim(N,M);
            matrixFlatten = new T[N * M];
            for (int i = 0;i < N;i++)
                for (int j = 0;j < M;j++)
                    matrixFlatten[i * M + j] = val;

            raw2DmatrixCache = construct2Dfromflat(matrixFlatten,n,m);
        }
        Matrix(T* flattenArr,int N,int M) {
            SetDim(N,M);
            matrixFlatten = flattenArr;
            raw2DmatrixCache = construct2Dfromflat(matrixFlatten,n,m);
        }
        Matrix(T** raw2Dmatrix,int N,int M) {
            SetDim(N,M);
            raw2DmatrixCache = raw2Dmatrix;
            matrixFlatten = flattenArray(raw2Dmatrix);
        }
        T* getFlattenMatrix() {
            return matrixFlatten;
        }
        T** get2Dmatrix() {
            if (raw2DmatrixCache == nullptr)
                raw2DmatrixCache = construct2Dfromflat(matrixFlatten,n,m);
            return raw2DmatrixCache;
        }

        void set(uint x,uint y,T val) {
            checkValidID(x,y);
            matrixFlatten[x * m + y] = val;
            raw2DmatrixCache[x][y] = val;
        }

        T get(uint x,uint y) {
            checkValidID(x,y);
            return matrixFlatten[x * m + y];
        }
        dim3 GetDim() {
            return dim3(n,m,0);
        }
        Matrix softMax() {
            T* rawResult = new T[lenFlattenCache];

            memcpy(rawResult,matrixFlatten,sizeof(T) * lenFlattenCache);
            T maxElm = CallGPUmax(rawResult,lenFlattenCache);
            T sum = CallSum(rawResult,lenFlattenCache);
            CallGPUExpMinusMax(rawResult,lenFlattenCache,maxElm);
            CallGPUSoftmax(rawResult,lenFlattenCache,sum);
            return Matrix(rawResult,n,m);
        }
        Matrix convolution(const Matrix& kernel,int stride = 1) {
            if (kernel.n % 2 || kernel.m % 2) {
                throw std::runtime_error("* Cannot process kernel dimemsion % 2 == 1");
            }
            auto result =  CallGPUConvolution(flattenArray,n,m,kernel.getFlattenMatrix(),kernel.n,kernel.m,stride,stride);
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
            for (int i = 0;i < n;i++)
               freeArr(raw2DmatrixCache[i]);
            freeArr(raw2DmatrixCache);
        }
    private:
        T* matrixFlatten;
        T** raw2DmatrixCache;
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