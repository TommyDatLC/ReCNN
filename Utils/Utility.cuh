//
// Created by datdau on 10/4/25.
//

#ifndef RECNN_UTILITY_CUH
#define RECNN_UTILITY_CUH

const int DEFAULT_KERNEL_SIZE = 1024;

__device__ int getID() {
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    return idx;
}

void CaculateBlockAndThreadNumber(int lengthArr,int& block ,int& thread) {
    thread = lengthArr < DEFAULT_KERNEL_SIZE ? lengthArr : DEFAULT_KERNEL_SIZE;
    block =  (lengthArr + thread - 1) / thread;
}
__global__ void reduce_sum(float *input,float *output) {
    __shared__ float sdata[1024];
    int id = getID();
    int threadID = threadIdx.x;

    // copy input len shared memory
    sdata[id] = input[id];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadID < s) sdata[id] += sdata[id + s]; // Neu thread id nam trong khoang [0 ,s -1]
        // đồng bộ thread trong cùng 1 block
        __syncthreads();

    }
    // Sử dụng mỗi thread 0 để gán tổng cho 1 cụm
    if (threadID == 0) output[blockIdx.x] = sdata[0];
}
float GPUSum(float *input,int length)
{

    float *d_input,*d_output;
    int outputLength = length;
    cudaMallocManaged(&d_input,sizeof(float) * length);
    cudaMallocManaged(&d_output,sizeof(float) * length);
    // copy to device
    for (int i = 0;i < length;i++) {
        d_input[i] = input[i];
    }
    int blocks = length,thread = 0;
    while (outputLength > 1) {
        CaculateBlockAndThreadNumber(length,blocks,thread);
        reduce_sum<<< blocks , thread>>>(d_input,d_output);
        outputLength = blocks;
        cudaDeviceSynchronize();
        std::cout << d_output[0] << '\n';
    }


}

__global__ void ExpSum(float* arr,int count,float* res) {
    int idx = getID();
    if (idx >= count)
    {
        return;
    }
  //  atomicAdd(res + )
}




#endif //RECNN_UTILITY_CUH