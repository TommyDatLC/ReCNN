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

## Model with Convolution: 
This is the Architecture of the Network with Convolution layer:
<img width="300" height="700" alt="image" src="https://cdn.discordapp.com/attachments/860346551448371220/1436705635780657243/Screenshot_2025-11-08_201506.png?ex=691093e4&is=690f4264&hm=207c349a4bb60b9d4b72a3938fb4a263af5a36e157b022e4a609a48ff3f0cef3&" />
- Accuracy (with 16x16 images of Cat and Dog): 46%
- Confusion matrix: 
<img width="600" height="300" alt="image" src="https://cdn.discordapp.com/attachments/860346551448371220/1437091128556916816/Screenshot_2025-11-09_214648.png?ex=6911fae8&is=6910a968&hm=f897feae46d0bad8f354eab41c103983f8834f8b8479b1fc76846bec0a245539&" />

## Model without Convolution: 
This is the Architecture of the Network without Convolution layer:
<img width="300" height="700" alt="image" src="https://cdn.discordapp.com/attachments/860346551448371220/1437094284351508543/Screenshot_2025-11-09_215741.png?ex=6911fdd9&is=6910ac59&hm=dc98966528686cd334ba7ff36e57a62bc81a4f2022425538736ca23ae3713cc4&" />
- Accuracy (with 16x16 images of Cat and Dog): 55%
- Confusion matrix:
<img width="300" height="700" alt="image" src="https://media.discordapp.net/attachments/860346551448371220/1437094822077792286/Screenshot_2025-11-09_220135.png?ex=6911fe59&is=6910acd9&hm=9ea4510b780d7ccd2468173b2975af5a61d7c3c3535eecacffc149036269f7c8&=&format=webp&quality=lossless" />

## Conclusion
- We can see that in this case, the network with the convolution layer is overfitting. Because the convolution and maxpooling layer scale the image down to 4x4, too 
small to train. 
- The one without a convolution layer still has low accuracy because the image size 
is still tiny, and the image hasn’t been enhanced yet.
<img width="300" height="700" alt="image" src="https://media.discordapp.net/attachments/860346551448371220/1437095381757333564/Screenshot_2025-11-09_220349.png?ex=6911fede&is=6910ad5e&hm=5b3038d9cc748d16a6352bd447a0159b463e33c6e30846ffce3412ec88dafbd5&=&format=webp&quality=lossless" />

# Sources
## GitHub repository: 
https://github.com/TommyDatLC/ReCNN.git 
## Dataset Download link: 
+) Raw data: https://github.com/TommyDatLC/ReCNN/tree/layers/Dataset 
+) Splitted data into 70% train and 30% test:  https://github.com/TommyDatLC/ReCNN/tree/layers/dataset_split

