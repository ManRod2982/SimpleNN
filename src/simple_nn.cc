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

#include "simple_nn.hh"

#include <iostream>

// Implementation of SimpleNN
SimpleNN::SimpleNN(std::vector<int> arch) {
  if (arch.size() < 2) {
    throw std::runtime_error(
        "Invalid architecture! it must have at least 2 layers");
  }

  // Create biases and weights
  for (int layer = 1; layer < arch.size(); layer++) {
    bias.push_back(Eigen::VectorXd(arch[layer]).setRandom());
    // Organize the weights to make the multiplication easier
    // For instance for a 784 30 10 architecture the weights would be
    // 30x784 and 10x30
    // That way when multiplying Wa+b
    // 30x784 * 784x1 ---> 30x1
    // So each row of weights corresponds to the weights of each neuron in the
    // next layer
    weights.push_back(
        Eigen::MatrixXd(arch[layer], arch[layer - 1])
            .setRandom());  // Organize weights to make the multiplication
  }
#ifdef DEBUG
  for (auto weight : weights) {
    std::cout << "Weights dimensions: " << weight.rows() << " and "
              << weight.cols() << std::endl;
  }
  for (auto layer_bias : bias) {
    std::cout << "Bias dimensions: " << layer_bias.rows() << " and "
              << layer_bias.cols() << std::endl;
  }
#endif
}

Eigen::MatrixXd SimpleNN::forward_propagation(Eigen::VectorXd input) {
  // Go through the layers and compute the values
  Eigen::MatrixXd activation = input;
  for (int layer = 0; layer < bias.size(); layer++) {
    activation = weights[layer] * activation + bias[layer];
    activation = activation_func(activation);
  }

  return activation;
}

Eigen::MatrixXd SimpleNN::activation_func(Eigen::MatrixXd activation) {
  // Sigmoid function used
  return 1 / (1 + ((-activation.array()).exp()));
}

int SimpleNN::validate_data(std::vector<Eigen::VectorXd> validation_data,
                            std::vector<Eigen::VectorXd> validation_labels) {
  // Check that the validation function is valid
  if (!check_if_valid) {
    std::cout << "Invalid validation function!" << std::endl;
    return 0;
  }

  // Perform data validation
  int valid = 0;
  for (size_t i = 0; i < validation_data.size(); i++) {
    auto sample = validation_data[i];
    auto label = validation_labels[i];
    auto output = forward_propagation(sample);
    valid = check_if_valid(output, label) ? valid++ : valid;
  }

  std::cout << "Validation " << valid << "/" << validation_data.size()
            << std::endl;
  return valid;
}

void SimpleNN::train(std::vector<Eigen::VectorXd> training_data,
                     std::vector<Eigen::VectorXd> labels) {
  // Split data into validation set and training
  std::vector<Eigen::VectorXd> validation_data(
      training_data.end() - nn_config.validation, training_data.end());
  std::vector<Eigen::VectorXd> validation_labels(
      labels.end() - nn_config.validation, labels.end());

  training_data.resize(training_data.size() - nn_config.validation);
  labels.resize(labels.size() - nn_config.validation);

#ifdef DEBUG
  std::cout << "Validation data size: " << validation_data.size() << std::endl;
  std::cout << "Validation label size: " << validation_labels.size()
            << std::endl;
  std::cout << "Training data size: " << training_data.size() << std::endl;
  std::cout << "labels size: " << labels.size() << std::endl;
#endif

  // Perform data validation before trainig
  (void)validate_data(validation_data, validation_labels);

  // Perform the number of epochs
  int epoch = 0;
  while (epoch < nn_config.epochs) {
    std::cout << "Epoch: " << epoch << std::endl;
    // run_mini_batches();
    (void)validate_data(validation_data, validation_labels);
    epoch++;
  }
}