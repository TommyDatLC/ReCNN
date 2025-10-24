//
// Created by datdau on 10/4/25.
//

#ifndef RECNN_UTILITY_CUH
#define RECNN_UTILITY_CUH


const int DEFAULT_KERNEL_SIZE = 1024;

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " -> " \
        << cudaGetErrorString(e) << " (" << e << ")\n"; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)
__device__ int getID() {
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    return idx;
}

void CaculateBlockAndThreadNumber(int lengthArr,int& block ,int& thread,int numthread = 0) {
    if (!numthread)
        numthread = DEFAULT_KERNEL_SIZE;
    thread = lengthArr < numthread ? lengthArr : numthread;
    block =  (lengthArr + thread - 1) / thread;
}
template <typename T>
__global__ void GPUreduce_sum(T *input,T *output) {
    __shared__ T sdata[1024];
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

template <typename T>
T CallSum(T *input,int length)
{

    T *d_input,*d_output;
    int outputLength = length;
    cudaMallocManaged(&d_input,sizeof(T) * length);
    cudaMallocManaged(&d_output,sizeof(T) * length);
    // copy to device
    for (int i = 0;i < length;i++) {
        d_input[i] = input[i];
    }
    int blocks = length,thread = 0;
    while (outputLength > 1) {
        CaculateBlockAndThreadNumber(length,blocks,thread);
        GPUreduce_sum<<< blocks , thread>>>(d_input,d_output);
        outputLength = blocks;
        cudaDeviceSynchronize();
    }
    return d_output[0];

}
template <typename T>
void freeArr(T* arr) {
    if (arr != nullptr)
        delete[] arr;
}



#endif //RECNN_UTILITY_CUH