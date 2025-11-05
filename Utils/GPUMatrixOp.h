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
__global__ void matrixMulTile(const T*  A,
                              const T*  B,
                              T* __restrict__ C,
                              int N, int K, int M) {
    constexpr int TILE = 16; // the complier will set this for you,not runtime
    __shared__ T tileA[TILE][TILE];
    __shared__ T tileB[TILE][TILE];

    int tx = threadIdx.x; // col within tile
    int ty = threadIdx.y; // row within tile
    int row = blockIdx.y * TILE + ty; // global row in C
    int col = blockIdx.x * TILE + tx; // global col in C

    T acc = 0;
    int numPhases = (K + TILE - 1) / TILE;
    for (int ph = 0; ph < numPhases; ++ph) {
        int aCol = ph * TILE + tx; // column to read from A
        int bRow = ph * TILE + ty; // row to read from B

        // load A[row, aCol] -> tileA[ty][tx]
        if (row < N && aCol < K) {
            tileA[ty][tx] = A[row * K + aCol];
        } else {
            tileA[ty][tx] = 0;
        }

        // load B[bRow, col] -> tileB[ty][tx]
        if (bRow < K && col < M) {
            tileB[ty][tx] = B[bRow * M + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();


        // multiply accumulate
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < M) {
        C[row * M + col] = acc;
    }
}



template <typename T>
T* CallMatrixMul(T* A, T* B, int HangA, int CotA, int CotB) {
    constexpr int TILE = 16; // phải khớp với kernel TILE
    T* h_outp = nullptr;
    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_outp = nullptr;

    // allocate host output
    try {
        h_outp = new T[(size_t)HangA * (size_t)CotB];
    } catch (...) {
        // bad_alloc
        return nullptr;
    }

    d_a = MallocAndCopyToDevice(A, (size_t)HangA * (size_t)CotA);
    if (!d_a) { delete[] h_outp; return nullptr; }

    d_b = MallocAndCopyToDevice(B, (size_t)CotA * (size_t)CotB);
    if (!d_b) { cudaFree(d_a); delete[] h_outp; return nullptr; }

    // allocate device output
    size_t outBytes = sizeof(T) * (size_t)HangA * (size_t)CotB;
    cudaError_t err = cudaMalloc((void**)&d_outp, outBytes);
    if (err != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); delete[] h_outp;
        return nullptr;
    }

    // launch kernel: block must match TILE
    dim3 block(TILE, TILE);
    dim3 grid( (CotB + TILE - 1) / TILE, (HangA + TILE - 1) / TILE );
    matrixMulTile<T><<<grid, block>>>(d_a, d_b, d_outp, HangA, CotA, CotB);

    // check launch & sync
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_outp); delete[] h_outp;
        return nullptr;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_outp); delete[] h_outp;
        return nullptr;
    }

    // copy back
    err = cudaMemcpy(h_outp, d_outp, outBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_outp); delete[] h_outp;
        return nullptr;
    }

    // free device memory
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
T* CallGPUmatrixBasicOP( T*  A, T2* B,int n,char OP) {
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
RawMatrixOutput<T> CallGPUConvolution(T* A
    ,int size3D,int N,int M,
    Tker* kernel,
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
    A[id] = randn * stddev; // no offset or
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

template <typename T>
__global__ void GPUTranspose3D(T* __restrict__ A, T* __restrict__ B,
                               int size3D, int n, int m) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // thứ tự: grid.x chạy theo cột (m)
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // grid.y chạy theo hàng (n)
    int s   = blockIdx.z * blockDim.z + threadIdx.z;  // slice

    if (s >= size3D || row >= n || col >= m) return;

    // A index: (s, row, col) với layout row-major từng slice: s * (n*m) + row*m + col
    // B index: (s, col, row) => s * (m*n) + col*n + row
    B[(size_t)s * (m * n) + (size_t)col * n + row] =
        A[(size_t)s * (n * m) + (size_t)row * m + col];
}
template <typename T>
T* CallGPUTranspose(T* A, int size3D, int n, int m) {
    if (A == nullptr) return nullptr;

    size_t in_len  = (size_t)size3D * n * m;
    size_t out_len = (size_t)size3D * m * n;

    // Device buffers
    T* d_A = nullptr;
    T* d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(T) * in_len));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(T) * out_len));

    // Copy input -> device
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(T) * in_len, cudaMemcpyHostToDevice));

    // block/grid: block 16x16, grid covers (m cols, n rows), grid.z = size3D
    dim3 block(16, 16, 1);
    dim3 grid( (m + block.x - 1) / block.x,
               (n + block.y - 1) / block.y,
               size3D ); // mỗi slice một layer z

    // Launch
    GPUTranspose3D<<<grid, block>>>(d_A, d_B, size3D, n, m);

    // Kiểm tra lỗi launch và đồng bộ
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B);
        return nullptr;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result về host
    T* B_host = new T[out_len];
    CUDA_CHECK(cudaMemcpy(B_host, d_B, sizeof(T) * out_len, cudaMemcpyDeviceToHost));

    // Giải phóng GPU memory
    cudaFree(d_A);
    cudaFree(d_B);

    return B_host; // caller phải delete[] sau khi dùng
}


#endif //RECNN_GPUMATRIXMUL_H