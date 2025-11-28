/*
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <iostream>

#include "mnist_reader.hh"
#include "simple_nn.hh"

int main() {
  // Read MNIST data and create a SimpleNN instance
  auto data = mnist_reader("data/train-images.idx3-ubyte",
                           "data/train-labels.idx1-ubyte");
  // Initialize Neural Network with
  // 784 Input layer (image is 28*28 = 784)
  // 30 Hidden layer
  // 10 Ouput layer
  std::cout << "Creating SimpleNN\n" << std::endl;
  SimpleNN nn({784, 30, 10});
  auto result = nn.forward_propagation(data.images[0]);
  std::cout << "Result: " << std::endl;
  std::cout << result << std::endl;
  std::cout << "Target: " << std::endl;
  std::cout << data.labels[0] << std::endl;
  nn.train(data.images, data.labels);
  return 0;
}
