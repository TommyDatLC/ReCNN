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
__global__ void GPUConvBackward(TAffect* pixelAffected,TAffect* convInp,T* WeightFromNextLayer,T* kernelWeight,
    dim3 convInpDim,dim3 weightFromNextLayerDim,
int inChannel,int kerSize,float learningRate,
T* outputWeight) {
    int ids = getIDx();
    int idx = getIDy();
    int idy = getIDz();
    if (ids >= convInpDim.x || idx >= convInpDim.y || idy >= convInpDim.z)
        return;
    int thisWeight = WeightFromNextLayer[ids * kerSize * kerSize + idx * kerSize + idy];
    dim3 pixelAffectId = pixelAffected[ids * kerSize * kerSize + idx * kerSize + idy].traceBackID;
    if (pixelAffectId.x == INT_MAX) // Cannot traceback (while using relu...)
        return;
    for (int i = 0;i < kerSize;i++) {
        int idxOffset = (pixelAffectId.y - kerSize / 2 + i);
        if (idxOffset > convInpDim.y)
            break;
        for (int j = 0;j < kerSize;j++) {
            int idyOffset  = pixelAffectId.z - kerSize / 2 + j;
            if (idyOffset > convInpDim.z)
                break;
            // bat dau tinh toan
            int idKernel = pixelAffectId.x * kerSize * kerSize + i * kerSize + j;
            int idActvation = (pixelAffectId.x % inChannel) * convInpDim.x * convInpDim.y  + idxOffset * convInpDim.y + idyOffset;
            T oldKernel = kernelWeight[idKernel];
            kernelWeight[idKernel] -= learningRate * thisWeight *  convInp[idActvation];
            outputWeight[idActvation] = convInp[idActvation] - learningRate * oldKernel * thisWeight;
        }
    }
}
template <typename T,typename TAffect = Tracebackable<T>>
T* CallGPUConvBackward(TAffect* pixelAffected,TAffect* convInp,T* weightFromNextLayer,T* kernelWeight,
    dim3 convInpDim,dim3 weightFromNextLayerDim,
int inChannel,int kerSize,float learningRate
) {
    int convInpLen = convInpDim.x * convInpDim.y * convInpDim.z;
    T* d_outputWeight;
    T* h_outputWeight = new T[convInpLen];
    cudaMalloc(&d_outputWeight, sizeof(TAffect) * convInpLen);
    TAffect* d_pixelAffected =  MallocAndCopyToDevice(pixelAffected,weightFromNextLayerDim);
    TAffect* d_convInp = MallocAndCopyToDevice(convInp,convInpDim);
    T* d_weightFromNextLayer = MallocAndCopyToDevice(weightFromNextLayer,weightFromNextLayerDim);
    T* d_kernelWeight = MallocAndCopyToDevice(kernelWeight,weightFromNextLayerDim.x * kerSize * kerSize);

    dim3 blocks,threads;
    Caculate3DBlockAndThread(convInpDim,blocks,threads );
    GPUConvBackward<<<blocks,threads>>>(d_pixelAffected,d_convInp,d_weightFromNextLayer,d_kernelWeight,convInpDim,weightFromNextLayerDim,inChannel,kerSize,learningRate,d_outputWeight);
    CopyToHost(h_outputWeight,d_outputWeight,convInpLen);
    return h_outputWeight;
}
#endif //RECNN_CONVOLUTIONLAYERBACKWARDDATA_H