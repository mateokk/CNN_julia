## Convolutional Neural Network in Julia
The goal of this project is to create custom implementation of CNN in Julia language to recognsie digits from the MNIST database without using any additional machine learning libraries. 
It's a simple network with custom parameters, but the main reason for building it was familiarise myself with neural network architectures and the principles of automatic differentiation.
This network consists of:
- **convolutional layer** with multiple kernels
- **maxpool layer** 
- **sigmoid layer**
- **flatten layer**
- **one hidden layer**
- **ReLu layer**
- **output layer**
- **softmax layer**

It trains itself image by image. The validation accuracy after 5 epochs on 5000 training images is 83,14%

![image](https://github.com/mateokk/CNN_julia/assets/132949097/97b386f0-37b2-4c88-b72c-314d35b1d1ae)




