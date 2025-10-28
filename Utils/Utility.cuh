//
// Created by datdau on 10/4/25.
//

#ifndef RECNN_UTILITY_CUH
#define RECNN_UTILITY_CUH


const int DEFAULT_KERNEL_SIZE = 1024;
#define ull unsigned long long
#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " -> " \
        << cudaGetErrorString(e) << " (" << e << ")\n"; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)
__device__ int getIDx() {
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    return idx;
}
__device__ int getIDy() {
    auto idx = threadIdx.y + blockDim.y * blockIdx.y;
    return idx;
}
__device__ int getIDz() {
    auto idx = threadIdx.z + blockDim.z * blockIdx.z;
    return idx;
}
template <typename T>
__device__ void swapDevice(T &a, T &b) {
    T temp = a;
    a = b;
    b = temp;
}
template <typename T>
void freeArr(T* arr) {
    if (arr != nullptr)
        delete[] arr;

}


void CaculateBlockAndThreadNumber(int lengthArr,int& block ,int& thread,int numthread = 0) {
    if (!numthread)
        numthread = DEFAULT_KERNEL_SIZE;
    thread = lengthArr < numthread ? lengthArr : numthread;
    block =  (lengthArr + thread - 1) / thread;

}
void Caculate3DBlockAndThread(int x,int y,int z,dim3& inpBlock,dim3& inpThread) {
    dim3 blocks((x + 9) / 10,(y + 9) / 10,(z + 9) / 10);
    dim3 threads(10,10,10);
    inpBlock = blocks;
    inpThread = threads;
}

template <typename T>
struct RawMatrixOutput {
    T* newRawMatrix;
    int Size3D,N,M;
};
        // trả về số channel
#endif //RECNN_UTILITY_CUH