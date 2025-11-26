# Simple Neural Network

Simple neural network implementation example on C++. The main example creates a Neural Network and trains it on the MNIST data set to detect hand written digits.

## Usage

To build this project:
```
mkdir build
cmake -B build
cmake --build build
```
To run the tests:
```
cd build
ctest
```

## Dependencies

This code uses Catch2 for testing as well as Cmake:

```
sudo apt install cmake
sudo apt install catch2
sudo apt-get install libeigen3-dev
```

To get the training, validation and testing data:

The MNIST dataset is in binary form and described [here](https://yann.lecun.org/exdb/mnist/) but basically for labels the file contains a:
- 32 bit magic number
- 32 bit number of labels
- unsigned bytes with the label from 0 to 9.

For images it is similar:
- 32 bit magic number
- 32 bit number of image
- 32 bit rows
- 32 bit columns
- unsigned byte pixels, row major.

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

Data was obtained [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and copied into `data` folder

And MNIST parser was created to read the data from binary into C++ std::vectors.

## Notes

These are just some personal notes:
- Gradient descent requires calculating the gradient over all the training samples and then updating the parameters (weights and biases in this case), stocastic gradient descent does it over a single sample randomly selected from the training set and the middle ground commonly used and implemented here is mini-batches, where a small sample of randomly selected samples is used, the average gradient is calculated over them and then we update.
- After finishing all the mini-batches we have completed an 'epoch', so for instance in a training set of 100 samples with a mini-batch of 10, we would update the gradient 10 times (split the 100 samples into 10 mini-batches and update after calculating the gradient on each) and that would be an 'epoch'.
- We define a 'Cost function' with the objective of minimizing it and hence reduce the error, we define it as follows:

$C(w,b) = \dfrac{1}{2n}\sum_{x}\|\mathbf{y}(x)-\mathbf{a}\|^2$

- $w$ denotes all our weights
- $b$ all our biases
- $n$ is our training sample
- $x$ is our training input and the sumation is over all our training inputs
- $y$ is our expected target, or our label in this case
- $a$ is the output from the network

Now with a way to define our target function to minimize the goal is to calculate the [gradient](https://en.wikipedia.org/wiki/Gradient) which will give us a way to find the local minima by updating against it, e.g if the gradient is going 'up' we want to go down and viceversa.

$-\eta\nabla C$

Where:
- $\eta$ is the learning rate, a parameter that will help us decide how 'fast' we want to move through the gradient
- $\nabla C$ is the gradient of the cost function.

However what we really want is the gradient of the cost function with respect to our parameters $w$ and $b$ since we want to modify those and have it reduce our cost function.

$w \to w'= w -\eta\frac{\partial{C}}{\partial{w}}$

$b \to b'= b -\eta\frac{\partial{C}}{\partial{b}}$

And for stochastic gradient descent with mini-batches we want to update our parameters over the average of the gradient of the mini-batch training sample so:

$w \to w'= w -\frac{\eta}{m}\sum_{m}\frac{\partial{C}}{\partial{w}}$

$b \to b'= b -\frac{\eta}{m}\sum_{m}\frac{\partial{C}}{\partial{b}}$

## References

Useful resources used in understanding and creating this:
- [Michael Nielsen's book on neural networks](http://neuralnetworksanddeeplearning.com/index.html)
- [3Blue1Brown Series on Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [IBM article on SGD](https://www.ibm.com/think/topics/stochastic-gradient-descent)
- [CS231n Andrej Karpathy's lecture on backpropagation](youtube.com/watch?v=i94OvYb6noo&list=PLrR1Mq4E9nirxSBJKtN1u4337Je3-xt6s&index=1&pp=gAQBiAQB0gcJCRUKAYcqIYzv)
- [Andrej Karpathy on why you should understand backpropagation](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)