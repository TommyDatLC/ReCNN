//
// Created by datdau on 10/14/25.
//

#ifndef RECNN_GPUPREFIXSUM_H
#define RECNN_GPUPREFIXSUM_H
#include "Utility.cuh"
__global__ void PrefixSum(float* input,int len) {
    __shared__ float shareMem[1024];

    auto lid = threadIdx.x;
    if (lid < len)
        shareMem[lid] = input[lid];
    else
        shareMem[lid] = 0;
    __syncthreads();

    // construct the tree
    for (int offset = 1;offset < len;offset <<= 1) {
        int id = (lid + 1) * offset * 2 - 1;
        if (id < len)
            shareMem[id] += shareMem[id - offset];
        __syncthreads();
    }

    shareMem[len - 1] = 0;
    __syncthreads();
    // inverse the tree here
    for (int offset = len >> 1 ;offset > 0;offset >>= 1) {
        int id = (lid + 1) * offset * 2 - 1;
        if (id < len) {
            float t = shareMem[id - offset]; // save the left one in T
            shareMem[id - offset] = shareMem[id]; //  the left = right
            shareMem[id] += t;// the right = right + old_left
        }
    }
        __syncthreads();

    if (lid < len)
        input[lid] = shareMem[lid] ;
}

void CallPrefixSum(float *input,int len) {
    float *d_input;
    cudaMalloc(&d_input,sizeof(float) * len);
    cudaMemcpy(d_input,input,sizeof(float) * len,cudaMemcpyHostToDevice);

    int numBlock,numThread;
    CaculateBlockAndThreadNumber(len,numBlock,numThread);
    PrefixSum<<<numBlock,numThread>>>(d_input,len);
    cudaMemcpy(input,d_input,sizeof(float) * len,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
#endif //RECNN_GPUPREFIXSUM_H

/*__global__ void prefix_sum(float *data) {
__shared__ float temp[1024];
int tid = threadIdx.x;

// Load data vào shared memory
temp[tid] = data[tid];
__syncthreads();

// Upsweep
for (int offset = 1; offset < blockDim.x; offset <<= 1) {
int i = (tid + 1) * offset * 2 - 1;
if (i < blockDim.x)
temp[i] += temp[i - offset];
__syncthreads();
}

// Đặt phần tử cuối thành 0 để chuẩn bị downsweep
if (tid == 0)
temp[blockDim.x - 1] = 0;
__syncthreads();

// Downsweep
for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
int i = (tid + 1) * offset * 2 - 1;
if (i < blockDim.x) {
float t = temp[i - offset];
temp[i - offset] = temp[i];
temp[i] += t;
}
__syncthreads();
}

// Ghi kết quả ra global
data[tid] = temp[tid];
}
*/