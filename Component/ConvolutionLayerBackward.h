//
// Created by datdau on 10/30/25.
//

#ifndef RECNN_CONVOLUTIONLAYERBACKWARDDATA_H
#define RECNN_CONVOLUTIONLAYERBACKWARDDATA_H
#include "Matrix.h"
using namespace TommyDat;
struct ConvBackwardData {
    Matrix<Tracebackable<float>>* ptr_nextLayerPixelAffect = nullptr;
    Matrix<float>* weight = nullptr;
};
// TracebackWeight
template <typename T,typename TAffect = Tracebackable<T>>
__global__ void GPUConvBackward(TAffect* pixelAffected,TAffect* convInp,T* WeightFromNextLayer,
    T* kernelWeight,T* outputKernelWeight,
    dim3 convInpDim,dim3 weightFromNextLayerDim,
int inChannel,int kerSize,float learningRate,
T* outputWeight) {
    int ids = getIDx();
    int idx = getIDy();
    int idy = getIDz();
    if (ids >= weightFromNextLayerDim.x || idx >= weightFromNextLayerDim.y || idy >= weightFromNextLayerDim.z)

        return;
    int idPixelAffect = ids * weightFromNextLayerDim.y * weightFromNextLayerDim.z + idx * weightFromNextLayerDim.z + idy;
    T thisWeight = WeightFromNextLayer[idPixelAffect];
    auto pixelAffect = pixelAffected[idPixelAffect];
    short pixelAffectx = pixelAffect.traceBackIDx;
    short pixelAffecty = pixelAffect.traceBackIDy;
    short pixelAffectz = pixelAffect.traceBackIDz;
    if (pixelAffectx == -1) // Cannot traceback (while using relu...)
        return;
    for (int i = 0;i < kerSize;i++) {
        int idxOffset = (pixelAffecty - kerSize / 2 + i);
        if (idxOffset >= convInpDim.y)
            break;
        for (int j = 0;j < kerSize;j++) {
            int idyOffset  = pixelAffectz - kerSize / 2 + j;
            if (idyOffset >= convInpDim.z)
                break;
            // bat dau tinh toan
            int idKernel = pixelAffectx * kerSize * kerSize + i * kerSize + j;
            int idActvation = (pixelAffectx % inChannel) * convInpDim.y * convInpDim.z  + idxOffset * convInpDim.z + idyOffset;
            T oldKernel = kernelWeight[idKernel];
            atomicAdd(&outputKernelWeight[idKernel], -learningRate * thisWeight *  convInp[idActvation]);
            atomicAdd(&outputWeight[idActvation],learningRate * oldKernel * thisWeight);
        }
    }
}
template <typename T,typename TAffect = Tracebackable<T>>
T* CallGPUConvBackward(TAffect* pixelAffected,TAffect* convInp,T* weightFromNextLayer,T* kernelWeight,
    dim3 convInpDim,dim3 weightFromNextLayerDim,
int inChannel,int kerSize,float learningRate
) {
    int convInpLen = convInpDim.x * convInpDim.y * convInpDim.z;
    T* h_outputWeight = new T[convInpLen];
    for (int i = 0;i < convInpLen;i++) {
        h_outputWeight[i] = 0;
    }
    T* d_outputWeight = MallocAndCopyToDevice(h_outputWeight,convInpLen);
    TAffect* d_pixelAffected =  MallocAndCopyToDevice(pixelAffected,weightFromNextLayerDim);
    TAffect* d_convInp = MallocAndCopyToDevice(convInp,convInpDim);
    T* d_weightFromNextLayer = MallocAndCopyToDevice(weightFromNextLayer,weightFromNextLayerDim);

    int KernelLen = weightFromNextLayerDim.x * kerSize * kerSize ;
    T* d_kernelWeight = MallocAndCopyToDevice(kernelWeight,KernelLen);
    T* d_outputKernel = MallocAndCopyToDevice(kernelWeight,KernelLen);

    dim3 blocks,threads;
    Caculate3DBlockAndThread(weightFromNextLayerDim,blocks,threads );
    CUDA_CHECK(cudaGetLastError());

    GPUConvBackward<<<blocks,threads>>>(d_pixelAffected,d_convInp,d_weightFromNextLayer,
        d_kernelWeight,d_outputKernel,
        convInpDim,weightFromNextLayerDim,
        inChannel,kerSize,learningRate,
        d_outputWeight);
    CopyToHost(h_outputWeight,d_outputWeight,convInpLen);
    CopyToHost(kernelWeight,d_outputKernel,KernelLen);
    cudaFree(d_convInp);
    cudaFree(d_kernelWeight);
    cudaFree(d_pixelAffected);
    cudaFree(d_weightFromNextLayer);
    return h_outputWeight;
}
#endif //RECNN_CONVOLUTIONLAYERBACKWARDDATA_H