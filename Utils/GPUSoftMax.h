//
// Created by datdau on 10/24/25.
//

#ifndef RECNN_GPUSOFTMAX_H
#define RECNN_GPUSOFTMAX_H
template <typename T>
__global__ void GPUExpMinusMax(T* input,T max) {
    int idx = getIDx();
    input[idx] = expf(input[idx] - max);
}
template <typename T>
__global__ void GPUSoftmax(T* input,T sum) {
    int idx = getIDx();
    input[idx] /= sum;
}
template <typename T>
void CallGPUSoftmax(T* inp,int length,T sum) {
    int lenInbyte = length * sizeof(T);
    T* d_inp;
    cudaMalloc(&d_inp,lenInbyte);
    cudaMemcpy(d_inp,inp,lenInbyte,cudaMemcpyHostToDevice);
    int block,thread;
    CaculateBlockAndThreadNumber(lenInbyte,block,thread);
    GPUSoftmax<<<block,thread>>>(d_inp,sum);
    cudaMemcpy(inp,d_inp,lenInbyte,cudaMemcpyDeviceToHost);
    cudaFree(d_inp);
}

template <typename T>
void CallGPUExpMinusMax(T* inp,int length,T max) {
    int lenInbyte = length * sizeof(T);
    T* d_inp;
    cudaMalloc(&d_inp,lenInbyte);
    cudaMemcpy(d_inp,inp,lenInbyte,cudaMemcpyHostToDevice);
    int block,thread;
    CaculateBlockAndThreadNumber(lenInbyte,block,thread);
    GPUExpMinusMax<<<block,thread>>>(d_inp,max);
    cudaMemcpy(inp,d_inp,lenInbyte,cudaMemcpyDeviceToHost);
    cudaFree(d_inp);
}
#endif //RECNN_GPUSOFTMAX_H