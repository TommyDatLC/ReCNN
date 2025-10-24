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
void freeArr(T* arr) {
    if (arr != nullptr)
        delete[] arr;
}



#endif //RECNN_UTILITY_CUH