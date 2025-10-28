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
__host__ T* flattenArray(T* arr,int size3D,int n,int m) {
    T* res = new T[n * m * size3D];
    for (int s = 0;s < size3D;s++)
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
        int aCol = ph * TILE + tx;     // column index in A to load
        int bRow = ph * TILE + ty;     // row index in B to load
        tileA[ty][tx] = (row < N && aCol < K) ? A[row * K + aCol] : 0;
        tileB[ty][tx] = (bRow < K && col < M) ? B[bRow * M + col] : 0;
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
__global__ void MatrixAddOrSub(T *A,T *B,T *C,int n,bool isAdd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n)
        return;
    if (isAdd)
        C[x] = A[x] + B[x];
    else
        C[x] = A[x] - B[x];
}
template <typename T>
T* CallMatrixAddOrSub(T* A,T *B,int n,bool isAdd) {
    T* d_A,*d_B;
    T* d_outp,*h_outp = new T[n];
    int allocSize = sizeof(T) * n;
    cudaMalloc(&d_A,allocSize);
    cudaMalloc(&d_B, allocSize);
    cudaMalloc(&d_outp,allocSize);
    cudaMemcpy(d_A,A,allocSize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,allocSize,cudaMemcpyHostToDevice);
    int blocks,threads;
    CaculateBlockAndThreadNumber(n,blocks,threads);
    MatrixAddOrSub<<<blocks,threads>>>(d_A,d_B,d_outp,n,isAdd);

    cudaMemcpy(h_outp,d_outp,allocSize,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_outp);
    return h_outp;
}
// assume that kernel is small
template <typename T>
__global__ void GPUConvolution(T* A,T* output,int size3D,int N,int M,T* kernel,int ks,int kn,int km,int stride,T sumOfKernel) {
    // Kernel always small so no need to use tile and share memory
    T value = 0;
    int inChannel = size3D,outChannel = ks;
    int ids = getIDx();
    int idx = getIDy();
    int idy = getIDz();
    if (ids >= ks)
        return;
        for (int i = 0;i < kn;i++)
            for (int j = 0;j < km;j++)
            {
                int idxWoffset  = (idx * stride - kn / 2) + i;
                int idyWoffset = (idy * stride - km / 2) + j;
                if (idxWoffset < 0 || idyWoffset < 0  || idxWoffset >= N || idyWoffset >= M)
                    continue;
                value += kernel[ids * km * kn + i * km + j] * A[(ids % inChannel) * M * N + (idxWoffset) * M + (idyWoffset )] ;
            }
    if (idx  <  N && idy < M) {
        output[ids * N * M + idx * M + idy] = value / sumOfKernel;
    }
}

template <typename T>
RawMatrixOutput<T> CallGPUConvolution(T* A,int size3D,int N,int M,T* kernel,int ks,int kn,int km,int stride) {
    int lenKer = kn * km * ks;
    int inChannel = size3D,outChannel = ks;
    T sumOfKernel = CPUsum(kernel,lenKer);
    int outputS = (ks + stride - 1) / stride;
    int outputN = (N  + stride - 1) / stride;
    int outputM = (M  + stride - 1) / stride;
    int lenOutput = outputS * outputM * outputN;

    T* d_A,*d_kernel,*d_output,*h_output = new T[lenOutput];
    int allocSize = sizeof(T) * N * M * size3D;
    cudaMalloc(&d_A,allocSize);
    cudaMemcpy(d_A,A,allocSize,cudaMemcpyHostToDevice);

    cudaMalloc(&d_output,sizeof(T) * lenOutput);
    cudaMemcpy(d_output,A,sizeof(T) * lenOutput,cudaMemcpyHostToDevice);

    cudaMalloc(&d_kernel,sizeof(T) * lenKer);
    cudaMemcpy(d_kernel,kernel,sizeof(T) * lenKer,cudaMemcpyHostToDevice);
    dim3 blocks,threads;
    Caculate3DBlockAndThread(outChannel,N,M,blocks,threads);
    GPUConvolution<<<blocks,threads>>>(d_A,d_output,size3D,N,M,d_kernel,ks,kn,km,stride,sumOfKernel);

    cudaMemcpy(h_output,d_output,sizeof(T) * lenOutput,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_kernel);
    return { h_output,outputS,outputN,outputM };
}
template <typename T>
__global__ void GPURelu(T *A,int n) {
    int id = getIDx();
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
    int id = getIDx();
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
__global__ void GPUmaxPooling(T* inputFlatten,T* output,int size3D,int n,int m,int outN,int outM,int size,int stride) {
    // sync for value into share mem
    int ids = getIDx();
    int idx = getIDy();
    int idy = getIDz();
    if (idx >= outN || idy >= outM || ids >= size3D)
        return;
    int idxOffsetStride = idx * stride;
    int idyOffsetStride = idy * stride;
    T max = -3e19;

        for (int i = 0;i < size;++i) {
                int idyCheck = idxOffsetStride + i;
                if (idyCheck >= n)
                    break;
            for (int j = 0;j < size;++j) {
                int idzCheck = idyOffsetStride + j;

                if (idzCheck >= m)
                    break;
                T check = inputFlatten[ ids * m * n + idyCheck  * m + idzCheck];
                if (check > max) {
                    max = check;
                }
            }
        }
        output[ ids * outM * outN +  idx * outM + idy] = max;
}

template <typename T>
RawMatrixOutput<T> CallGPUmaxPooling(T* A,int size3D,int n,int m,int size,int stride) {
    T* d_A;
    int len = size3D * n * m;
    cudaMalloc(&d_A,sizeof(T) * len );
    cudaMemcpy(d_A,A,sizeof(T) * len,cudaMemcpyHostToDevice);

    int outN = (n + stride - 1) / stride;
    int outM = (m + stride - 1) / stride;
    dim3 blocks,threads;
    Caculate3DBlockAndThread(size3D,outN,outM,blocks,threads);

    int lenOut = size3D *  outN  * outM;

    T* h_output = new T[lenOut],*d_output;
    cudaMalloc(&d_output,sizeof(T) *  lenOut);
    GPUmaxPooling<<<blocks,threads>>>(d_A,d_output,size3D,n,m,outN,outM,size,stride);
    cudaMemcpy(h_output,d_output,sizeof(T) * lenOut,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_output);

    return { h_output, size3D, outN, outM };
}
template <typename T>
__global__ void GPUNormalize(T* A,int n,T maxN) {
    int id = getIDx();
    if (id < n)
        A[id] /= maxN;
}
template <typename T>
void CallGPUNormalize(T* A,int n,T maxN) {
    T* d_a;
    cudaMalloc(&d_a,sizeof(T) * n);
    cudaMemcpy(d_a,A,sizeof(T) * n,cudaMemcpyHostToDevice);

    int blocks,threads;
    CaculateBlockAndThreadNumber(n,blocks,threads);
    GPUNormalize<<<blocks,threads>>>(d_a,n,maxN);
    cudaMemcpy(A,d_a,sizeof(T) * n,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}
template <typename T,typename TdeviceFunc>
__global__ void GPUapply(T* A,int n,TdeviceFunc f) {
    int id = getIDx();
    if (id < n)
        A[id] = f(A[id]);
}
template <typename T,typename TdeviceFunc>
void CallGPUapply(T* A,int n,TdeviceFunc f) {
    T* d_a;
    cudaMalloc(&d_a,sizeof(T) * n);
    cudaMemcpy(d_a,A,sizeof(T) * n,cudaMemcpyHostToDevice);

    int blocks,threads;
    CaculateBlockAndThreadNumber(n,blocks,threads);
    GPUapply<<<blocks,threads>>>(d_a,n,f);
    cudaMemcpy(A,d_a,sizeof(T) * n,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}
template <typename T>
__global__ void GPUTranspose(T* A,int size3D,int n,int m) {
    int s = getIDx();
    int x = getIDy();
    int y = getIDz();
    if (s >= size3D || x >= n)
        return;
    swapDevice(A[s * n * m + x * m + y],A[s * n * m + y * m + x]);
}
template <typename T>
void CallGPUTranspose(T* A,int size3D,int n,int m) {
    T* d_a;
    int len = size3D * n * m;
    cudaMalloc(&d_a,sizeof(T) * len);
    cudaMemcpy(d_a,A,sizeof(T) * len,cudaMemcpyHostToDevice);

    dim3 blocks,threads;
    Caculate3DBlockAndThread(size3D,n,m,blocks,threads);
    GPUTranspose<<<blocks,threads>>>(d_a,size3D,n,m);
    cudaMemcpy(A,d_a,sizeof(T) * len,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}
#endif //RECNN_GPUMATRIXMUL_H