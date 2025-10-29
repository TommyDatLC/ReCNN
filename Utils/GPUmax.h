//
// Created by datdau on 10/13/25.
//

#ifndef RECNN_GPUMAX_H
#define RECNN_GPUMAX_H
#include "Utility.cuh"

template <typename T>
__global__ void GPUmax(T* inputArr,T* blockOutputArr) {
    __shared__ float shareMem[1024];
    int id = getIDx();
    int tid = threadIdx.x;

    shareMem[tid] = inputArr[id];
    __syncthreads();

    for (int s = blockDim.x / 2 ; s > 0; s >>= 1) {
        if (tid < s)
           shareMem[id] = fmaxf(shareMem[id],shareMem[id + s]);
        __syncthreads();
    }
    if (tid == 0) blockOutputArr[blockIdx.x] = shareMem[0];
}
template <typename T>
int CallGPUmax(T* input,int length) {
    T* d_input,*d_output;
    cudaMallocManaged(&d_input,sizeof(T) * length);
    cudaMallocManaged(&d_output,sizeof(T) * length);
    int block ,thread;
    cudaMemcpy(d_input,input,sizeof(T) * length,cudaMemcpyHostToDevice);
    while (length > 1) {
        CaculateBlockAndThreadNumber(length,block,thread);
        GPUmax<<<block,thread>>>(d_input,d_output);
        length = block;
        cudaDeviceSynchronize();
    }
    return d_output[0];
}

#endif //RECNN_GPUMAX_H