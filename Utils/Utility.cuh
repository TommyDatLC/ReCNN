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
       \
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

template<typename T>
__device__ void swapDevice(T &a, T &b) {
    T temp = a;
    a = b;
    b = temp;
}

template<typename T>
void freeArr(T *arr) {
    if (arr != nullptr)
        delete[] arr;
}
template<typename T>
void CopyToDevice(T* ptr_host,T* ptr_device,int size) {
    cudaMemcpy( ptr_device, ptr_host, sizeof(T) * size, cudaMemcpyHostToDevice);
}
template<typename T>
void CopyToHost(T* ptr_host,T* ptr_device,int size) {
    cudaMemcpy(ptr_host , ptr_device, sizeof(T) * size, cudaMemcpyDeviceToHost);
}
template<typename T>
T* MallocAndCopyToDevice(T* ptr_host, int size) {
    T *ptr_device = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr_device, sizeof(T) * size));
    CopyToDevice(ptr_host,ptr_device,size);
    return ptr_device;
}

template<typename T>
T *MallocAndCopyToDevice(T *ptr_host, dim3 size) {
    int len = size.x * size.y * size.z;
    return MallocAndCopyToDevice(ptr_host, len);
}

void CaculateBlockAndThreadNumber(int lengthArr, int &block, int &thread,int numthread = 0) {
    if (!numthread)
        thread = DEFAULT_KERNEL_SIZE;
    else
        thread = lengthArr < numthread ? lengthArr : numthread;
    block = (lengthArr + thread - 1) / thread;
}

void Caculate3DBlockAndThread(int x, int y, int z, dim3 &inpBlock, dim3 &inpThread) {
    dim3 blocks((x + 9) / 10, (y + 9) / 10, (z + 9) / 10);
    dim3 threads(10, 10, 10);
    inpBlock = blocks;
    inpThread = threads;
}
void Caculate3DBlockAndThread(dim3 size,dim3& inpBlock,dim3& inpThread) {
    Caculate3DBlockAndThread(size.x,size.y,size.z,inpBlock,inpThread);
}

template<typename T>
struct RawMatrixOutput {
    T *newRawMatrix;
    int Size3D, N, M;
};


// trả về số channel
#endif //RECNN_UTILITY_CUH
