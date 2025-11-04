//
// Created by datdau on 10/30/25.
//

#ifndef RECNN_CONVOLUTIONLAYERBACKWARDDATA_H
#define RECNN_CONVOLUTIONLAYERBACKWARDDATA_H
#include "Matrix.h"
using namespace TommyDat;
struct ConvBackwardData {
    Matrix<Tracebackable<float>>* ptr_nextLayerPixelAffect = nullptr;
    Matrix<float>* gradient = nullptr;
    // ~ConvBackwardData() {
    //     delete gradient;
    //     delete ptr_nextLayerPixelAffect;
    // }
};
// TracebackWeight
template <typename T,typename TAffect = Tracebackable<T>>
__global__ void GPUConvBackward(TAffect* pixelAffected,TAffect* convInp,T* GradFromNextLayer,
    T* kernelWeight,T* outputKernelWeight,
    dim3 convInpDim,dim3 gradientDim,
int inChannel,int kerSize,float learningRate,
T* outputGrad) {
    int ids = getIDx();
    int idx = getIDy();
    int idy = getIDz();
    if (ids >= gradientDim.x || idx >= gradientDim.y || idy >= gradientDim.z)

        return;
    int idPixelAffect = ids * gradientDim.y * gradientDim.z + idx * gradientDim.z + idy;
    T thisGrad = GradFromNextLayer[idPixelAffect];
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
            atomicAdd(&outputKernelWeight[idKernel], -learningRate * thisGrad *  convInp[idActvation]);
            atomicAdd(&outputGrad[idActvation],oldKernel * thisGrad);
        }
    }
}
template <typename T,typename TAffect = Tracebackable<T>>
T* CallGPUConvBackward(TAffect* pixelAffected,TAffect* convInp,T* gradient,T* kernelWeight,
    dim3 convInpDim,dim3 gradientDim,
int inChannel,int kerSize,float learningRate
) {
    int convInpLen = convInpDim.x * convInpDim.y * convInpDim.z;
    T* h_outputGrad = new T[convInpLen];
    for (int i = 0;i < convInpLen;i++) {
        h_outputGrad[i] = 0;
    }
    T* d_outputWeight = MallocAndCopyToDevice(h_outputGrad,convInpLen);
    TAffect* d_pixelAffected =  MallocAndCopyToDevice(pixelAffected,gradientDim);
    TAffect* d_convInp = MallocAndCopyToDevice(convInp,convInpDim);
    T* d_weightFromNextLayer = MallocAndCopyToDevice(gradient,gradientDim);

    int KernelLen = gradientDim.x * kerSize * kerSize ;
    T* d_kernelWeight = MallocAndCopyToDevice(kernelWeight,KernelLen);
    T* d_outputKernel = MallocAndCopyToDevice(kernelWeight,KernelLen);
    dim3 blocks,threads;
    Caculate3DBlockAndThread(gradientDim,blocks,threads );
    cudaGetLastError();

    GPUConvBackward<<<blocks,threads>>>(d_pixelAffected,d_convInp,d_weightFromNextLayer,
        d_kernelWeight,d_outputKernel,
        convInpDim,gradientDim,
        inChannel,kerSize,learningRate,
        d_outputWeight);

    CopyToHost(h_outputGrad,d_outputWeight,convInpLen);
    CopyToHost(kernelWeight,d_outputKernel,KernelLen);

    cudaFree(d_convInp);
    cudaFree(d_kernelWeight);
    cudaFree(d_pixelAffected);
    cudaFree(d_weightFromNextLayer);
    cudaFree(d_outputWeight);
    cudaFree(d_outputKernel);
    return h_outputGrad;
}
#endif //RECNN_CONVOLUTIONLAYERBACKWARDDATA_H