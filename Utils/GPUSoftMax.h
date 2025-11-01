//
// Created by datdau on 10/24/25.
//

#ifndef RECNN_GPUSOFTMAX_H
#define RECNN_GPUSOFTMAX_H
template <typename T>
__global__ void GPUExpMinusMax(T* input,int len,T max) {
    int idx = getIDx();
    if (idx >= len)
        return;
    input[idx] = expf(input[idx] - max);
}
template <typename T>
__global__ void GPUSoftmax(T* input,int len,T sum) {
    int idx = getIDx();
    if (idx >= len)
        return;
    input[idx] /= sum;
}
template <typename T>
void CallGPUSoftmax(T* inp,int length,T sum) {

    T* d_inp = MallocAndCopyToDevice(inp,length);

    int block,thread;
    CaculateBlockAndThreadNumber(length,block,thread);
    GPUSoftmax<<<block,thread>>>(d_inp,length,sum);

    CopyToHost(inp,d_inp,length);
    cudaFree(d_inp);
}

template <typename T>
void CallGPUExpMinusMax(T* inp,int length,T max) {

    T* d_inp = MallocAndCopyToDevice(inp,length);
    int block,thread;
    CaculateBlockAndThreadNumber(length,block,thread);
    GPUExpMinusMax<<<block,thread>>>(d_inp,length,max);
    CopyToHost(inp,d_inp,length);
    cudaFree(d_inp);
}
#endif //RECNN_GPUSOFTMAX_H