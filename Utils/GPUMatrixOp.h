//
// Created by datdau on 10/22/25.
//

#ifndef RECNN_GPUMATRIXMUL_H
#define RECNN_GPUMATRIXMUL_H
#include "GPUmax.h"
#include "Sum.h"
#include "Utility.cuh"
#include "../Component/Tracebackable.h"


template <typename T>
__host__ __device__ T get2D(const T* __restrict__ arr2Dflatten,int x,int y,int n,int m) {
    if (x >= n || y >= m)
        return 0;
    return arr2Dflatten[x * m + y];
}
template <typename T>
__host__ __device__  void set2D(T* __restrict__ arr2Dflatten,int x,int y,int n,int m,T val) {
    arr2Dflatten[x * m + y] = val;
}
template <typename T>
__host__ T* flattenArray(const T* __restrict__ arr,int size3D,int n,int m) {
    T* res = new T[n * m * size3D];
    for (int s = 0;s < size3D;s++)
    for (int i = 0;i < n;i++)
        for (int j = 0;j < m;j++) {
            res[i * m + j] = arr[i][j];

        }
    return res;
}
template <typename T>
T** construct2Dfromflat(T* __restrict__ arr,int n,int m) {
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
__global__ void matrixMulTile(const T* A, const T* B, T* __restrict__ C, int N, int K, int M) {
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

        if (row < N && aCol < K)
            tileA[ty][tx] = A[row * K + aCol];
        else
            tileA[ty][tx] = 0;

        if (row < N && aCol < K)
            tileA[ty][tx] = B[bRow * M + col];
        else
            tileA[ty][tx] = 0;

        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += tileA[ty][k] * tileB[k][tx];
        __syncthreads();
    }
    if (row < N && col < M) C[row * M + col] = acc;
}


template <typename T>
T* CallMatrixMul(T* A, T* B,int HangA,int CotA,int CotB) {
    T* d_outp,*h_outp = new T[HangA * CotB];
    T* d_a =    MallocAndCopyToDevice(A,HangA * CotA);
    T* d_b =  MallocAndCopyToDevice(B,CotA * CotB);

    cudaMalloc(&d_outp,sizeof(T) * HangA * CotB);
    dim3 block(32, 32);
    dim3 grid( (CotB + 31) / 32, (HangA + 31) / 32 ); // round-up division
    matrixMulTile<<<grid,block>>>(d_a, d_b, d_outp,HangA, CotA, CotB);

    // kiểm tra lỗi kernel và đồng bộ
    CUDA_CHECK(cudaGetLastError()); // catch launch error
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(h_outp,d_outp,sizeof(T) * HangA * CotB,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_outp);
    return h_outp;
}
#define MAT_OP_ADD 0
#define MAT_OP_SUBTRACT 1
#define MAT_OP_MUL 2
template <typename T,typename T2>
__global__ void GPUmatrixBasicOP(T* __restrict__ A,const T2* __restrict__ B,int n,char Operation) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n)
        return;
    switch (Operation) {
        case MAT_OP_ADD:
            A[x] += B[x];
            break;
        case MAT_OP_SUBTRACT:
            A[x] -= B[x];
            break;
        case MAT_OP_MUL:
            A[x] *= B[x];

    }
}
template <typename T,typename T2>
T* CallGPUmatrixBasicOP( T* __restrict__ A, T2* __restrict__ B,int n,char OP) {
    T* d_A;
    T2 * d_B;
    T* h_outp = new T[n];

   d_A = MallocAndCopyToDevice(A,n);
   d_B = MallocAndCopyToDevice(B,n);

    int blocks,threads;
    CaculateBlockAndThreadNumber(n,blocks,threads);
    GPUmatrixBasicOP<<<blocks,threads>>>(d_A,d_B,n,OP);

    CopyToHost(h_outp,d_A,n);
    cudaFree(d_A);
    cudaFree(d_B);

    return h_outp;
}
// assume that kernel is small
template <typename T,typename Tker>
__global__ void GPUConvolution(
    const T* __restrict__ A,
    int size3D,int N,int M,
          T* __restrict__  output,
    int OutputS,int OutputN,int OutputM,
    const Tker* __restrict__ kernel,
    int ks,int kn,int km,
    int stride) {
    // Kernel always small so no need to use tile and share memory
    T value = 0;
    int inChannel = size3D,outChannel = ks;
    int ids = getIDx();
    int idx = getIDy();
    int idy = getIDz();
    int idxOffsetStride = idx * stride;
    int idyOffsetStride = idy * stride;
    if (ids >= OutputS)
        return;
        for (int i = 0;i < kn;i++)
            for (int j = 0;j < km;j++)
            {
                int idxWoffset  = (idxOffsetStride - kn / 2) + i;
                int idyWoffset = (idyOffsetStride - km / 2) + j;
                if (idxWoffset < 0 || idyWoffset < 0  || idxWoffset >= N || idyWoffset >= M)
                    continue;
                value += A[(ids % inChannel) * M * N + (idxWoffset) * M + (idyWoffset )] * kernel[ids * km * kn + i * km + j]  ;
            }
    if (idx  <  OutputN && idy < OutputM) {
        int id = ids * OutputN * OutputM + idx * OutputM + idy;
       // int idOffset = ids * N * M + idxOffsetStride * M + idyOffsetStride;

        output[id] = value;
        if constexpr (std::is_same_v<T,TommyDat::Tracebackable<float>>) {
            output[id].traceBackIDx = ids;
            output[id].traceBackIDy = idxOffsetStride;
            output[id].traceBackIDz = idyOffsetStride;
        }
    }
}

template <typename T,typename Tker>
RawMatrixOutput<T> CallGPUConvolution(T* __restrict__ A
    ,int size3D,int N,int M,
    Tker* __restrict__ kernel,
    int ks,int kn,int km,int stride) {
    int lenKer = kn * km * ks;
    int inChannel = size3D,
        outChannel = ks;
    int allocSize = N * M * size3D;

    int outputS = (ks);
    int outputN = (N  + stride - 1) / stride;
    int outputM = (M  + stride - 1) / stride;
    int lenOutput = outputS * outputM * outputN;

    Tker* d_kernel = MallocAndCopyToDevice(kernel,lenKer);
    T* d_A =  MallocAndCopyToDevice(A,allocSize);

    T *d_output,*h_output = new T[lenOutput];
    cudaMalloc(&d_output,sizeof(T) * lenOutput);
    dim3 blocks,threads;
    Caculate3DBlockAndThread(outChannel,N,M,blocks,threads);
    GPUConvolution<<<blocks,threads>>>(d_A,size3D,N,M,
                                        d_output,outputS,outputN,outputM,
                                        d_kernel,ks,kn,km,
                                        stride);

    cudaDeviceSynchronize();
    CopyToHost(h_output,d_output,lenOutput);
    cudaFree(d_A);
    cudaFree(d_kernel);
    cudaFree(d_output);
    return { h_output,outputS,outputN,outputM };
}
template <typename T>
__global__ void GPURelu(T *A,int n) {
    int id = getIDx();
    if (id >= n) return;
    if (A[id] < 0) {
        A[id] = 0;
        if constexpr (std::is_same_v<T,TommyDat::Tracebackable<float>>) {
              A[id].traceBackIDx = -1;
        }
    }
}
template <typename T>
void CallGPURelu(T* __restrict__ A,int n) {
    T* d_a;
    cudaMalloc(&d_a,sizeof(T) * n);
    cudaMemcpy(d_a,A,sizeof(T) * n,cudaMemcpyHostToDevice);
    // might not optimize for memory coalesing
    int blocks,threads;
    CaculateBlockAndThreadNumber(n,blocks,threads);
    GPURelu<<<blocks,threads>>>(d_a,n);
    cudaMemcpy(A,d_a,sizeof(T) * n,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}
#include <curand_kernel.h>
template <typename T>
__global__ void GPUheInit(T* __restrict__ A,int n,int in,ull seed) {
    int id = getIDx();
    if (id >=  n)
        return;
    curandState state;
    curand_init(seed,id,0,&state);
    T randn = curand_normal(&state);

    T stddev = sqrtf(2.f/ in);
    A[id] = randn * stddev;
}
template <typename T>
void CallGPUheInit(T* __restrict__ A,int n,int in,ull seed) {
    T* d_a;
    cudaMalloc(&d_a,sizeof(T) * n);
    cudaMemcpy(d_a,A,sizeof(T) * n,cudaMemcpyHostToDevice);

    int blocks,threads;
    CaculateBlockAndThreadNumber(n,blocks,threads);

    GPUheInit<<<blocks,threads>>>(d_a,n,in,seed);
    cudaMemcpy(A,d_a,sizeof(T) * n,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}
template <typename T>
__global__ void GPUmaxPooling(T* __restrict__ inputFlatten,T* __restrict__ output,int size3D,int n,int m,int outN,int outM,int size,int stride) {
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
RawMatrixOutput<T> CallGPUmaxPooling(T* __restrict__ A,int size3D,int n,int m,int size,int stride) {
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
__global__ void GPUNormalize(T* __restrict__ A,int n,T maxN) {
    int id = getIDx();
    if (id < n)
        A[id] /= maxN;
}
template <typename T>
void CallGPUNormalize(T* __restrict__ A,int n,T maxN) {
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
void CallGPUapply(T* __restrict__ A,int n,TdeviceFunc f) {
    T* d_a;
    cudaMalloc(&d_a,sizeof(T) * n);
    cudaMemcpy(d_a,A,sizeof(T) * n,cudaMemcpyHostToDevice);

    int blocks,threads;
    CaculateBlockAndThreadNumber(n,blocks,threads);
    GPUapply<<<blocks,threads>>>(d_a,n,f);
    cudaMemcpy(A,d_a,sizeof(T) * n,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}
// Kernel: out-of-place transpose for each slice s.
// Input layout (flatten): A[ s * (n*m) + x * m + y ]  where x in [0..n-1], y in [0..m-1]
// Output layout (flatten): B[ s * (m*n) + y * n + x ]  -> shape (size3D, m, n)
template <typename T>
__global__ void GPUTranspose3D(const T* __restrict__ A, T* __restrict__ B,
                               int size3D, int n, int m) {
    int s = blockIdx.z * blockDim.z + threadIdx.z;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (s >= size3D || x >= n || y >= m)
        return;

    // Gán phần tử (s, x, y) ở A sang (s, y, x) ở B
    B[s * m * n + y * n + x] = A[s * n * m + x * m + y];
}

template <typename T>
T* CallGPUTranspose(T* A, int size3D, int n, int m) {
    int in_len = size3D * n * m;
    int out_len = size3D * m * n;

    // Cấp phát và copy input lên device
    T* d_A = MallocAndCopyToDevice(A, in_len);

    // Cấp phát vùng nhớ cho output trên device
    T* d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(T) * out_len));

    // Cấu hình block & grid
    dim3 threads;
    dim3 blocks;
    Caculate3DBlockAndThread(size3D,n,m,blocks,threads);
    // Gọi kernel
    GPUTranspose3D<<<blocks, threads>>>(d_A, d_B, size3D, n, m);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cấp phát vùng nhớ host output
    T* B_host = new T[out_len];

    // Copy từ device về host
    CopyToHost(B_host, d_B, out_len);

    // Giải phóng GPU memory
    cudaFree(d_A);
    cudaFree(d_B);

    // Trả về con trỏ host chứa ma trận đã transpose
    return B_host;
}

#endif //RECNN_GPUMATRIXMUL_H