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

#ifndef SIMPLE_NN_HH
#define SIMPLE_NN_HH
#include <eigen3/Eigen/Dense>
#include <vector>

// Declaration of SimpleNN
class SimpleNN {
 public:
  // Ctor of SimpleNN that expects a vector containing the
  // architecture of the Neural Network, each entry in the vector
  // corresponds to a layer for instance
  // [80 30 10]
  // would be a NN with 80 neurons in the input layer
  // 30 neurons in the hidden layer
  // 10 neurons in the output layer
  SimpleNN(std::vector<int> arch);
  ~SimpleNN() = default;
  // Configuration parameters for training
  struct training_config {
    int epochs = 30;  // Number of epochs to be trained, default value of 30
    double learning_rate =
        3.0;              // Learning rate to be used, default value of 3.0
    int mini_batch = 10;  // Mini-batch size to be used, default value of 10
  };

  // Method used to trigger the training of the neural network
  // It takes:
  // - Vector of vectors containing the training data
  // - Vector of vectors containing the labels for the training data
  // - A configuration structure with the hyper-parameters used in the training
  void train(std::vector<Eigen::VectorXd> training_data,
             std::vector<Eigen::VectorXd> labels, training_config config);

  // Method used for Forward propagation
  // it returns the result of the output layer
  Eigen::MatrixXd forward_propagation(Eigen::VectorXd input);

  // Add neural network APIs here
 private:
  // Activation function
  Eigen::MatrixXd activation_func(Eigen::MatrixXd activation);
  // Weights of the Neural Network
  std::vector<Eigen::MatrixXd> weights;
  // Biases of the Neural Network
  std::vector<Eigen::VectorXd> bias;
};

#endif  // SIMPLE_NN_HH
