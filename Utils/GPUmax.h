//
// Created by datdau on 10/13/25.
//

#ifndef RECNN_GPUMAX_H
#define RECNN_GPUMAX_H
#include "Utility.cuh"



__global__ void device_max(float* inputArr,float* blockOutputArr) {
    __shared__ float shareMem[1024];
    int id = getID();
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

int GPUmax(float* input,float length) {
    float* d_input,*d_output;
    cudaMallocManaged(&d_input,sizeof(float) * length);
    cudaMallocManaged(&d_output,sizeof(float) * length);
    int block ,thread;
    cudaMemcpy(d_input,input,sizeof(float) * length,cudaMemcpyHostToDevice);

    while (length > 1) {
        CaculateBlockAndThreadNumber(length,block,thread);
        device_max<<<block,thread>>>(d_input,d_output);
        length = block;
        cudaDeviceSynchronize();
    }
    return d_output[0];
}
#endif //RECNN_GPUMAX_H