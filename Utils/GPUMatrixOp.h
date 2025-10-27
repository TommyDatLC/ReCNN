//
// Created by datdau on 10/22/25.
//

#ifndef RECNN_GPUMATRIXMUL_H
#define RECNN_GPUMATRIXMUL_H
#include "GPUmax.h"
#include "Sum.h"
#include "Utility.cuh"

template <typename T>
__host__ __device__ T get2D(T* arr2Dflatten,int x,int y,int n,int m) {
    if (x >= n || y >= m)
        return 0;
    return arr2Dflatten[x * m + y];
}
template <typename T>
__host__ __device__  void set2D(T* arr2Dflatten,int x,int y,int n,int m,T val) {
    arr2Dflatten[x * m + y] = val;
}
template <typename T>
__host__ T* flattenArray(T* arr,int n,int m) {
    T* res = new T[n * m];
    for (int i = 0;i < n;i++)
        for (int j = 0;j < m;j++) {
            res[i * m + j] = arr[i][j];

        }
    return res;
}
template <typename T>
T** construct2Dfromflat(T* arr,int n,int m) {
    T** result = new T*[n];
    for (int i = 0;i < n;i++) {
        result[i] = new T[m];
        for (int j =0; j < m;j++) {
            result[i][j] = arr[i * n + j];
        }
    }
    return result;
}

template <typename T>
__global__ void matrixMulTile(const T* A, const T* B, T* C, int N, int K, int M) {
    constexpr int TILE = 32;
    __shared__ T tileA[TILE][TILE];
    __shared__ T tileB[TILE][TILE];

    int tx = threadIdx.x; // col within tile
    int ty = threadIdx.y; // row within tile
    int row = blockIdx.y * TILE + ty; // global row
    int col = blockIdx.x * TILE + tx; // global col

    T acc = static_cast<T>(0);
    int numPhases = (K + TILE - 1) / TILE;
    for (int ph = 0; ph < numPhases; ++ph) {
        int a_col = ph * TILE + tx;     // column index in A to load
        int b_row = ph * TILE + ty;     // row index in B to load
        tileA[ty][tx] = (row < N && a_col < K) ? A[row * K + a_col] : 0;
        tileB[ty][tx] = (b_row < K && col < M) ? B[b_row * M + col] : 0;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += tileA[ty][k] * tileB[k][tx];
        __syncthreads();
    }
    if (row < N && col < M) C[row * M + col] = acc;
}


template <typename T>
T* CallMatrixMul(T *A,T *B,int HangA,int CotA,int CotB) {
    T* d_outp,*h_outp = new T[HangA * CotB];
    T* d_a;
    T* d_b;
    int* Debug;
    cudaMallocManaged(&Debug,sizeof(T) * 1024);
    cudaMalloc(&d_a,sizeof(T) * HangA * CotA);
    cudaMalloc(&d_b, sizeof(T) * CotB * CotA );
    cudaMalloc(&d_outp,sizeof(T) * HangA * CotB);
    cudaMemcpy(d_a,A,sizeof(T) * HangA * CotA,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,B,sizeof(T) * CotB * CotA,cudaMemcpyHostToDevice);
    dim3 block(32, 32);
    dim3 grid( (CotB + 31) / 32, (HangA + 31) / 32 ); // round-up division
    matrixMulTile<<<grid,block>>>(d_a, d_b, d_outp,HangA, CotA, CotB);

    // kiểm tra lỗi kernel và đồng bộ
    CUDA_CHECK(cudaGetLastError()); // catch launch error
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(h_outp,d_outp,sizeof(T) * CotB * CotA,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_outp);
    return h_outp;
}
template <typename T>
__global__ void MatrixAddOrSub(T *A,T *B,T *C,int Hang,int Cot,bool isAdd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int thisIdx = x * Cot + y;

    if (isAdd)
        C[thisIdx] = A[thisIdx] + B[thisIdx];
    else
        C[thisIdx] = A[thisIdx] - B[thisIdx];
}
template <typename T>
T* CallMatrixAddOrSub(T* A,T *B,int Hang,int Cot,bool isAdd) {
    T* d_A,*d_B;
    T* d_outp,*h_outp = new T[Hang * Cot];
    int allocSize = sizeof(T) * Hang * Cot;
    cudaMalloc(&d_A,allocSize);
    cudaMalloc(&d_B, allocSize);
    cudaMalloc(&d_outp,allocSize);
    cudaMemcpy(d_A,A,allocSize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,allocSize,cudaMemcpyHostToDevice);
    dim3 blocks((Hang + 31) / 32,(Cot + 31) / 32);
    dim3 threads(32,32);
    MatrixAddOrSub<<<blocks,threads>>>(d_A,d_B,d_outp,Hang,Cot,isAdd);
    cudaDeviceSynchronize();
    cudaMemcpy(h_outp,d_outp,allocSize,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_outp);
    return h_outp;
}
// assume that kernel is small
template <typename T>
__global__ void GPUConvolution(T* A,T* output,int N,int M,T* kernel,int kn,int km,int strideX,int strideY,T sumOfKernel) {
    // Kernel always small so no need to use tile and share memory
    T value = 0;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    for (int i = 0;i < kn;i++)
        for (int j = 0;j < km;j++)
        {
            int idxWoffset  = (idx * strideX - kn / 2) + i;
            int idyWOffset = (idy * strideY - km / 2) + j;
            if (idxWoffset < 0 || idyWOffset < 0 || idxWoffset >= N || idyWOffset >= M)
                continue;
            value += kernel[i * km + j] * A[(idxWoffset) * M + (idyWOffset )] ;
        }
    if (idx  <  N && idy < M) {
        output[idx * M + idy] = value / sumOfKernel;
    }
}
template <typename T>
struct RawMatrixOutput {
    T* newRawMatrix;
    int N,M;
};
template <typename T>
RawMatrixOutput<T> CallGPUConvolution(T* A,int N,int M,T* kernel,int kn,int km,int strideX = 1,int strideY = 1) {
    T sumOfKernel = CPUsum(kernel,kn * km);
    int outputN = (N  + strideX - 1) / strideX;
    int outputM = (M  + strideY - 1) / strideY;

    T* d_A,*d_kernel,*d_output,*h_output = new T[outputM * outputN];
    int allocSize = sizeof(T) * N * M;
    cudaMalloc(&d_A,allocSize);
    cudaMemcpy(d_A,A,allocSize,cudaMemcpyHostToDevice);


    cudaMalloc(&d_output,sizeof(T) * outputN * outputM);
    cudaMemcpy(d_output,A,sizeof(T) * outputN * outputM,cudaMemcpyHostToDevice);

    cudaMalloc(&d_kernel,sizeof(T) * kn * km);
    cudaMemcpy(d_kernel,kernel,sizeof(T) * kn * km,cudaMemcpyHostToDevice);

    dim3 blocks((N + 31) / 32,(M + 31) / 32);
    dim3 threads(32,32);
    GPUConvolution<<<blocks,threads>>>(d_A,d_output,N,M,d_kernel,kn,km,strideX,strideY,sumOfKernel);

    cudaMemcpy(h_output,d_output,sizeof(T) * outputN * outputM,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_kernel);
    return { h_output,outputN,outputM };
}
template <typename T>
__global__ void GPURelu(T *A,int n) {
    int id = getID();
    if (A[id] < 0 && id < n)
        A[id] = 0;
}
template <typename T>
void CallGPURelu(T *A,int n) {
    T* d_a;
    cudaMalloc(&d_a,sizeof(T) * n);
    cudaMemcpy(d_a,A,sizeof(T) * n,cudaMemcpyHostToDevice);

    int blocks,threads;
    CaculateBlockAndThreadNumber(n,blocks,threads);
    GPURelu<<<blocks,threads>>>(d_a,n);
    cudaMemcpy(A,d_a,sizeof(T) * n,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}
#include <curand_kernel.h>
__global__ void GPUheInit(float* A,int n,int in,ull seed) {
    int id = getID();
    if (id >=  n)
        return;
    curandState state;
    curand_init(seed,id,0,&state);
    float randn = curand_normal(&state);

    float stddev = sqrtf(2.f/ in);
    A[id] = randn * stddev;
}
void CallGPUheInit(float* A,int n,int in,ull seed) {
    float* d_a;
    cudaMalloc(&d_a,sizeof(float) * n);
    cudaMemcpy(d_a,A,sizeof(float) * n,cudaMemcpyHostToDevice);

    int blocks,threads;
    CaculateBlockAndThreadNumber(n,blocks,threads);

    GPUheInit<<<blocks,threads>>>(d_a,n,in,seed);
    cudaMemcpy(A,d_a,sizeof(float) * n,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}
template <typename T>
__global__ void GPUmaxPooling(T* inputFlatten,T* output,int n,int m,int outN,int outM,int size,int stride) {
    // sync for value into share mem
    int idx = threadIdx.x + blockDim.x  * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx >= outN || idy >= outM) return;

    int idxOffsetStride = idx * stride;
    int idyOffsetStride = idy * stride;

    T max = -3e19;

    for (int i = 0;i < size;++i) {
        int idxCheck = idxOffsetStride + i;

        if (idxCheck >= n)
            break;

        for (int j = 0;j < size;++j) {
            int idyCheck = idyOffsetStride + j;

            if (idyCheck >= m)
                break;

            T check = inputFlatten[idxCheck * m + idyCheck];
            if (check > max) {
                max = check;
            }
        }
    }

        output[idx * outM + idy] = max;
}
template <typename T>
RawMatrixOutput<T> CallGPUmaxPooling(T* A,int n,int m,int size,int stride) {
    T* d_A;
    cudaMalloc(&d_A,sizeof(T) * n * m);
    cudaMemcpy(d_A,A,sizeof(T) * n * m,cudaMemcpyHostToDevice);

    dim3 blocks((n + 31) / 32,(m + 31) / 32);
    dim3 threads(32,32);

    int outN = (n + stride - 1) / stride;
    int outM = (m + stride - 1) / stride;
    T* h_output = new T[outN  * outM],*d_output;
    cudaMalloc(&d_output,sizeof(T) *  outN * outM);

    GPUmaxPooling<<<blocks,threads>>>(d_A,d_output,n,m,outN,outM,size,stride);
    cudaMemcpy(h_output,d_output,sizeof(T) * outN * outM,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_output);

    return { h_output , outN, outM };
}
template <typename T>
 __global__ void GPUnormalize(T* input,int n,int maxNumber = 255) {

    int id = getID();
    if (id >= n)
        return;
    input[id] /= maxNumber;
}
template <typename T>
void CallGPUnormalize(T* input,int n, int maxNumber = 255) {
    T* d_input;
    cudaMalloc(&d_input,sizeof(T) * n);
    cudaMemcpy(d_input,input,sizeof(T) * n,cudaMemcpyHostToDevice);

    int blocks,threads;
    CaculateBlockAndThreadNumber(n,blocks,threads);
    GPUnormalize<<<blocks,threads>>>(input,n,maxNumber);
    cudaMemcpy(input,d_input,sizeof(T) * n,cudaMemcpyDeviceToHost);

    cudaFree(d_input);
}
#endif //RECNN_GPUMATRIXMUL_H