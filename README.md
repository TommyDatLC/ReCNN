<img width="620" height="127" alt="image" src="https://github.com/user-attachments/assets/73891fd3-db1a-4aca-91c0-1c216be243b7" />

# Introduction 
Re::CNN is a complete from-scratch implementation of Convolutional Neural 
Networks built entirely in C++/CUDA without using any high-level deep learning 
frameworks (TensorFlow, PyTorch, etc.). The project demonstrates: 
- Pure C++/CUDA Implementation: Every component is hand-coded, from 
basic matrix operations to complex backpropagation algorithms, using only 
standard C++ libraries and minimal dependencies. 
- Modular Architecture: Clean separation of concerns with distinct 
components: Layers, Base Classes, Neural Network, Activation Functions.
- High-Performance Computing: Matrix operations for convolution and backpropagation. 
Moreover, they support for CUDA acceleration (configurable in build system). 
- Educational Deep Dive: Complete transparency into CNN internals, 
making every mathematical operation visible and understandable.

# Input and Output descriptions
## Input
1. Raw RGB images of cats and dogs. 
2. Image dimensions: 16x16 (small) or 400x400 (large) pixels, configurable.  
3. Training dataset: Cat images (label: 0) and Dog images (label: 1). 
4. Training dataset structure : <img width="300" height="300" alt="image" src="https://media.discordapp.net/attachments/860346551448371220/1436702149324181534/Screenshot_2025-11-08_200045.png?ex=691090a5&is=690f3f25&hm=78937e529c1ac941180b498fb1ff4a212ed673e2ae1390106599ea52c1bbcbb1&=&format=webp&quality=lossless" />
5. Hyperparameters: learning rate, epochs, kernel sizes, and number of 
channels. 
6. Configuration options: activation functions and network architecture.

## Output
1. A trained binary classifier model (Cat vs Dog) 
2. Predictions for test images: probability distribution [P(cat), P(dog)] 
3. Classification decision: 0 (Cat) or 1 (Dog) based on highest probability 
4. Training metrics: 
+ Binary cross-entropy loss curves over iterations 
+ Classification accuracy progression through epochs 
+ Validation performance (accuracy, confusion matrix) 
5. Performance benchmarks: 
+ Training loss progression 
+ Per-epoch accuracy on test set 
6. Model checkpoints saved for future inference 
+ Serialize model’s architecture, layers, biases, weights and other 
hyperparameters into JSON file 
+ Autoloaded and saved for maximum training

# Obtained Outcomes Descriptions
## Complete CNN Architecture Implementation
1. ConvolutionLayer with configurable kernel size, stride, channels 
2.  MaxPoolingLayer for spatial dimension reduction 
3.  Fully Connected (FCLayer) for classification 
4.  Forward and backward propagation through all layer types  
5.  Auto gradient computation with tracebackable operations 
6.  Multiple activation functions (ReLU, Softmax)

## Training Infrastructure
1.  Loss computation and backpropagation through entire network 
2.  Weight initialization strategies 
3.  Model serialization/deserialization for saving trained models 
4.  Data loading pipeline with train/test split support

## Performance Metrics
We built 2 models with different architectures, one with a convolution layer, max 
pooling, and the second one without a Convolution layer, Max Pooling layer, with 
the learning rate equal to 0.001 


# Sources
## GitHub repository: 
https://github.com/TommyDatLC/ReCNN.git 
## Dataset Download link: 
- Raw data: https://github.com/TommyDatLC/ReCNN/tree/layers/Dataset 
- Splitted data into 70% train and 30% test:  https://github.com/TommyDatLC/ReCNN/tree/layers/dataset_split

# Project Flow
## Mathematical Foundation and Design 
- Objective: Understand CNN mathematics and design the architecture
- Activities:
  - Study convolution operation: output[i,j] = Σ Σ input[i+m,j+n] * kernel[m,n]
  - Understand gradient flow through conv layers using chain rule
  - Design class hierarchy (LayerBase → ConvolutionLayer, FCLayer, MaxPoolingLayer)
  - Plan data structures for Matrix and gradients (Matrix class)
- Key Files: LayerBase.h, Matrix.h, design documentation

## Core Layer Implementation
Objective: Build fundamental building blocks
- Convolution Layer -> ConvolutionLayer.h 
- Max Pooling Layer -> MaxPoolingLayer.h 
- Fully Connected Layer -> FClayer.h 
- Activation Function -> EnumActivationType.h

## Training Loop Implementation 
Objective: Train the network and evaluate the performance 
- Training and evaluate -> main16.cu

## Testing and Validation 
Objective: Evaluate test set, add track metrics, and early stopping if accuracy 
plateaus 
- Testing -> main16.cu

## Save and Load trained models 
Objective: Save and load trained models 
- Save and Load -> Serialize.h

## Optimization and Performance 
Objective: Accelerate computation using GPU kernel 
- Main key files:  
1. GPUMatrixOp.h  
2. GPUMax.h 
3. GPUPrefixSum.h 
4. GPUSoftMax.h

# Source Code Tutorial
- Step 1: Clone the repository  
- Step 2: Install nvcc, g++, gcc, Clion, and CMake on Linux 
(Linux required since it is not compatible with clang and gcc/g++ on Windows) 
Run on Clion 
- Step 3: To read, training and testing use: 
trainingData = ReadImage16x16(false); 
testData = ReadImage16x16(true); 
- Step 4: Call TrainAndEval() to evaluate and train the model 
- Step 5: To save the model after trained 
ModelSerialize::saveNetwork(*net,"../Models/mymodel.json"); 
- Step 6: To load the model for training 
ModelSerialize::loadNetwork<NeuralInput>("../Models/mymodel.json");
