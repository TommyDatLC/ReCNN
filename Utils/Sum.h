//
// Created by tommydatlc on 10/24/25.
//

#ifndef RECNN_SUM_H
#define RECNN_SUM_H
#include "Utility.cuh"


template <typename T>
__global__ void GPUreduce_sum(T *input,T *output,int len) {
    __shared__ T sdata[1024];
    int id =  getIDx();
    int lId = threadIdx.x;

    // copy input len shared memory
    if (lId >=  len)
        sdata[lId] = (T)0;
    else
        sdata[lId] = input[id];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lId < s)
            sdata[lId] += sdata[lId + s]; // Neu thread id nam trong khoang [0 ,s - 1]
        // đồng bộ thread trong cùng 1 block
        __syncthreads();
    }
    // Sử dụng mỗi thread 0 để gán tổng cho 1 cụm
    if (lId == 0)
        output[blockIdx.x] = sdata[0];
}

template <typename T >
T CallGPUSum(T *input,int length)
{

    T *h_output = new T[length];
    int outputLength = length;
    T *d_input = MallocAndCopyToDevice(input,length);
    T *d_output;
    cudaMalloc(&d_output,sizeof(T) * length);

    int thread = 1024;
    int blocks = (length + thread - 1) / thread;
    //CaculateBlockAndThreadNumber(outputLength,blocks,thread);
    GPUreduce_sum<<< blocks , thread>>>(d_input,d_output,outputLength);
    CUDA_CHECK(cudaGetLastError());
    outputLength = blocks;

    CopyToHost(h_output,d_output,length);
    cudaFree(d_input);
    cudaFree(d_output);

    return h_output[0];
}

template <typename T>
T CPUsum(T* arr,int len) {
    T res = 0.0f;
    for (int i =0 ;i < len;i++)
        res += arr[i];
    return res;
}

#endif //RECNN_SUM_H