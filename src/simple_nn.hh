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
  // Ctor of SimpleNN that expects a path to the saved parameters
  SimpleNN(const std::string &path);
  ~SimpleNN() = default;

  // Method used to set configuration hyper parameters
  // it takes:
  // - Number of epochs to be used
  // - The learning rate to be used
  // - The mini batch size to be used
  // - Size of validation set to be used from the training data set
  void set_config(int epochs, double learning_rate, int mini_batch,
                  int validation) {
    nn_config.epochs = epochs;
    nn_config.learning_rate = learning_rate;
    nn_config.mini_batch = mini_batch;
    nn_config.validation = validation;
  };

  // Method used to trigger the training of the neural network
  // It takes:
  // - Vector of vectors containing the training data
  // - Vector of vectors containing the labels for the training data
  void train(std::vector<Eigen::VectorXd> training_data,
             std::vector<Eigen::VectorXd> labels);

  // Method used for Forward propagation
  // it returns the result of the output layer
  Eigen::MatrixXd forward_propagation(Eigen::VectorXd input);

  // Set validation data function
  void set_validation_function(
      std::function<bool(Eigen::VectorXd, Eigen::VectorXd)> func) {
    check_if_valid = func;
  }

  // Method used to validate the data
  // a function to validate the data needs to be passed
  int validate_data(std::vector<Eigen::VectorXd> validation_data,
                    std::vector<Eigen::VectorXd> validation_labels);

  // Method used to save the weights and biases to a file
  // Takes the path to save the parameters to
  void save_parameters(const std::string &path);

  // Method used to load saved weights and biases from a file
  // Takes the path to load the parameters from
  void load_parameters(const std::string &path);

 private:
  // Function used to validate the data
  std::function<bool(Eigen::VectorXd, Eigen::VectorXd)> check_if_valid;

  // Methond used to run the mini-batches
  // it takes the training data and labels
  void run_mini_batches(std::vector<Eigen::VectorXd> training_data,
                        std::vector<Eigen::VectorXd> labels);

  struct deltas {
    std::vector<Eigen::MatrixXd> delta_w;
    std::vector<Eigen::VectorXd> delta_b;
  };

  // Method used to perform backpropagation and calculate the gradient
  // Requires an input and the expected output
  // Returns a structure with the gradient over weights and biases
  deltas backpropagation(Eigen::VectorXd input, Eigen::VectorXd output);

  // Activation function
  Eigen::MatrixXd activation_func(Eigen::MatrixXd activation);
  // Derivative of the activation function
  double activation_func_prime(double x);
  // Weights of the Neural Network
  std::vector<Eigen::MatrixXd> weights;
  // Biases of the Neural Network
  std::vector<Eigen::VectorXd> bias;

  // Configuration parameters for training
  struct training_config {
    int epochs = 30;  // Number of epochs to be trained, default value of 30
    double learning_rate =
        3.0;              // Learning rate to be used, default value of 3.0
    int mini_batch = 10;  // Mini-batch size to be used, default value of 10
    int validation = 0;   // Size of validation data set to be used from the
                          // training set, default of 0
  };
  // Training configuration structure
  training_config nn_config;
  // Magic number for parameters data
  static constexpr uint32_t param_data = 0xDEC1B092;
};

#endif  // SIMPLE_NN_HH
