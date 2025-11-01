//
// Created by datdau on 10/13/25.
//

#ifndef RECNN_GPUMAX_H
#define RECNN_GPUMAX_H
#include "Utility.cuh"

template <typename T>
__global__ void GPUmax(T* inputArr,T* blockOutputArr,int len) {
    __shared__ float shareMem[1024];
    int id = getIDx();
    int tid = threadIdx.x;
    if (id >= len)
        return;
    shareMem[tid] = inputArr[id];
    __syncthreads();

    for (int s = blockDim.x / 2 ; s > 0; s >>= 1) {
        if (tid < s)
           shareMem[tid] = fmaxf(shareMem[tid],shareMem[tid + s]);
        __syncthreads();
    }
    if (tid == 0)
        blockOutputArr[blockIdx.x] = shareMem[0];
}
template <typename T>
T CallGPUmax(T* input,int length) {
    T* d_input,*d_output;
    d_input = MallocAndCopyToDevice(input,length);
    CUDA_CHECK(cudaMalloc(&d_output,sizeof(T) * length));
    int block ,thread = 1024;
    int outLen = length;
    while (outLen > 1) {
        if (outLen != length)
            std::swap(d_input,d_output);
        block = (outLen + thread - 1) / thread;
        GPUmax<<<block,thread>>>(d_input,d_output,outLen);
        outLen = block;
        CUDA_CHECK(cudaGetLastError());
    }
    T* h_output = new T[length];
    CopyToHost(h_output,d_output,length);
    return h_output[0];
}

#endif //RECNN_GPUMAX_H