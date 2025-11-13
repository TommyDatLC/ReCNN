# Introduction 

<p align="center">
<img width="620" height="127" alt="image" src="https://github.com/user-attachments/assets/73891fd3-db1a-4aca-91c0-1c216be243b7" />
</p>

**Re::CNN** is a complete from-scratch implementation of Convolutional Neural Networks built entirely in **C++/CUDA** without using any high-level deep learning frameworks (TensorFlow, PyTorch, etc.).

The project demonstrates:

- **Pure C++/CUDA Implementation:** Every component is hand-coded, from basic matrix operations to complex backpropagation algorithms, using only standard C++ libraries and minimal dependencies.  
- **Modular Architecture:** Clean separation of concerns with distinct components — Layers, Base Classes, Neural Network, Activation Functions.  
- **High-Performance Computing:** Matrix operations for convolution and backpropagation with support for CUDA acceleration (configurable in the build system).  
- **Educational Deep Dive:** Complete transparency into CNN internals, making every mathematical operation visible and understandable.

---

# Obtained Outcomes Descriptions

## Complete CNN Architecture Implementation

<p align="center">
<img width="1177" height="660" alt="image" src="https://github.com/user-attachments/assets/23059dd6-7d8c-4eac-9434-45a92eba0e6f" />
</p>

1. ConvolutionLayer with configurable kernel size, stride, channels  
2. MaxPoolingLayer for spatial dimension reduction  
3. Fully Connected (FCLayer) for classification  
4. Forward and backward propagation through all layer types  
5. Auto gradient computation with tracebackable operations  
6. Multiple activation functions (ReLU, Softmax)

---

## Training Infrastructure

1. Loss computation and backpropagation through entire network  
2. Weight initialization strategies  
3. Model serialization/deserialization for saving trained models  
4. Data loading pipeline with train/test split support  

---

## Performance Metrics

Two models were built with different architectures:  
- One with a Convolution layer and Max Pooling  
- One without a Convolution or Max Pooling layer  
Both trained with a **learning rate = 0.001**

---

# Sources

## Dataset Download Links

- Raw data: [Dataset](https://github.com/TommyDatLC/ReCNN/tree/layers/Dataset)  
- Split data (70% train / 30% test): [dataset_split](https://github.com/TommyDatLC/ReCNN/tree/layers/dataset_split)

---

# Project Flow

## Mathematical Foundation and Design 

**Objective:** Understand CNN mathematics and design the architecture  

**Activities:**
- Study convolution operation:

$$
\text{output}[i, j] = \sum_m \sum_n \text{input}[i + m, j + n] \cdot \text{kernel}[m, n]
$$

- Understand gradient flow through convolution layers using the chain rule  
- Design class hierarchy (`LayerBase` → `ConvolutionLayer`, `FCLayer`, `MaxPoolingLayer`)  
- Plan data structures for `Matrix` and gradients (`Matrix` class)

**Key Files:**
``` text
1. GPUMatrixOp.h  
2. GPUMax.h 
3. GPUPrefixSum.h 
4. GPUSoftMax.h
```
# Source Code Tutorial
- Step 1: Clone the repository  
- Step 2: Install nvcc, g++, gcc, Clion, and CMake on Linux 
(Linux required since it is not compatible with clang and gcc/g++ on Windows) 
Run on Clion 
- Step 3: To read, training and testing use:
```text
trainingData = ReadImage16x16(false); 
testData = ReadImage16x16(true);
```
- Step 4: Call ```TrainAndEval() ``` to evaluate and train the model 
- Step 5: To save the model after trained
```text
ModelSerialize::saveNetwork(*net,"../Models/mymodel.json");
```
- Step 6: To load the model for training
```text
ModelSerialize::loadNetwork<NeuralInput>("../Models/mymodel.json");
```

