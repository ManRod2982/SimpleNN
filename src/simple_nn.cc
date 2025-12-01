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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

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

SimpleNN::SimpleNN(const std::string& path) {
  // Load parameters data from file
  load_parameters(path);
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

double SimpleNN::activation_func_prime(double x) {
  // Sigmoid function used
  auto s = 1.0 / (1.0 + std::exp(-x));
  return s * (1.0 - s);
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
    valid = check_if_valid(output, label) ? ++valid : valid;
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
    run_mini_batches(training_data, labels);
    (void)validate_data(validation_data, validation_labels);
    epoch++;
  }
}

void SimpleNN::run_mini_batches(std::vector<Eigen::VectorXd> training_data,
                                std::vector<Eigen::VectorXd> labels) {
  // Create indexes for mini batches
  std::vector<int> indexes(training_data.size());
  std::iota(indexes.begin(), indexes.end(), 0);

  // Shuffle training data
  // Initialize random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(indexes.begin(), indexes.end(), gen);

  // Run over mini-batches updating each time
  // Index is updated inside the loop
  for (size_t i = 0; i < indexes.size();) {
    std::vector<Eigen::MatrixXd> delta_w;
    std::vector<Eigen::VectorXd> delta_b;

    // Initialize sizes
    for (size_t layer = 0; layer < this->weights.size(); layer++) {
      delta_w.push_back(Eigen::MatrixXd::Zero(this->weights[layer].rows(),
                                              this->weights[layer].cols()));
      delta_b.push_back(Eigen::VectorXd::Zero(this->bias[layer].rows()));
    }

    // Train over mini batch size
    for (size_t mini = 0; mini < nn_config.mini_batch; mini++) {
      // Check boundary
      if (i >= indexes.size()) {
        break;
      }
      // Get shuffled index
      auto index = indexes[i];
      auto res = backpropagation(training_data[index], labels[index]);
      // Add new weights and biases for the average
      for (size_t layer = 0; layer < res.delta_w.size(); layer++) {
        delta_w[layer] += res.delta_w[layer];
        delta_b[layer] += res.delta_b[layer];
      }
      i++;
    }
    // Update weights and biases
    for (size_t layer = 0; layer < this->weights.size(); layer++) {
      double eta = nn_config.learning_rate / nn_config.mini_batch;
      this->weights[layer] -= eta * delta_w[layer];
      this->bias[layer] -= eta * delta_b[layer];
    }
  }
}

SimpleNN::deltas SimpleNN::backpropagation(Eigen::VectorXd input,
                                           Eigen::VectorXd output) {
  // Perform foward pass and save activation and z's
  std::vector<Eigen::MatrixXd> activations;
  std::vector<Eigen::MatrixXd> z;
  // On the last layer the input is basically the 'activation'
  activations.push_back(input);
  for (size_t layer = 0; layer < this->weights.size(); layer++) {
    z.push_back(this->weights[layer] * activations[layer] + this->bias[layer]);
    activations.push_back(activation_func(z[layer]));
  }

  // Compute gradients
  std::vector<Eigen::MatrixXd> delta_w;
  std::vector<Eigen::VectorXd> delta_b;

  // Perform backward pass
  // Only the first pass requires the cost function
  auto last_layer = activations.size() - 1;
  auto cost = activations[last_layer] - output;
  auto z_prime = z[last_layer - 1].unaryExpr(
      [this](double val) { return this->activation_func_prime(val); });
  Eigen::MatrixXd delta = (z_prime.array() * cost.array()).matrix();
  auto tmp_w = delta * activations[last_layer - 1].transpose();
  auto tmp_b = delta;
  delta_w.push_back(tmp_w);
  delta_b.push_back(tmp_b);

  for (size_t layer = weights.size() - 1; layer > 0; layer--) {
    auto z_prime = z[layer - 1].unaryExpr(
        [this](double val) { return this->activation_func_prime(val); });
    delta = weights[layer].transpose() * delta;
    delta = (delta.array() * z_prime.array()).matrix();
    auto tmp_w = delta * activations[layer - 1].transpose();
    auto tmp_b = delta;

    // Insert to the front since we are going backwards
    delta_w.insert(delta_w.begin(), tmp_w);
    delta_b.insert(delta_b.begin(), tmp_b);
  }
  return {delta_w, delta_b};
}

void SimpleNN::save_parameters(const std::string& path) {
  std::ofstream parameters(path, std::ios_base::out | std::ios_base::binary);
  if (parameters.is_open()) {
    // Write Magic number
    parameters.write(reinterpret_cast<const char*>(&param_data),
                     sizeof(param_data));
    // Write the number of layers
    uint32_t layers = weights.size();
    parameters.write(reinterpret_cast<const char*>(&layers), sizeof(layers));

    // Go through layers writing weights
    for (const auto& weight : weights) {
      // Write rows and cols
      uint32_t rows = weight.rows();
      uint32_t cols = weight.cols();
      parameters.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
      parameters.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
      // Write the weights data from the matrix
      parameters.write(reinterpret_cast<const char*>(weight.data()),
                       rows * cols * sizeof(Eigen::MatrixXd::Scalar));
    }

    // Go through layers writing bias
    for (const auto& b : bias) {
      // Write rows and cols
      uint32_t rows = b.rows();
      uint32_t cols = b.cols();
      parameters.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
      parameters.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
      // Write the weights data from the matrix
      parameters.write(reinterpret_cast<const char*>(b.data()),
                       rows * cols * sizeof(Eigen::VectorXd::Scalar));
    }

    parameters.close();
  } else {
    std::cerr << "Unable to open file" << std::endl;
  }
}

void SimpleNN::load_parameters(const std::string& path) {
  std::ifstream parameters(path, std::ios::in | std::ios::binary);
  if (!parameters.is_open()) {
    std::cerr << "Error: Could not open file " << path << " for reading."
              << std::endl;
  }

  // Check the Magic number
  uint32_t header;
  parameters.read(reinterpret_cast<char*>(&header), sizeof(header));

  if (header != param_data) {
    std::cerr << "Invalid parameter header!" << std::endl;
    return;
  }

  // Read the number of layers
  uint32_t layers;
  parameters.read(reinterpret_cast<char*>(&layers), sizeof(layers));
  std::cout << layers << " layers" << std::endl;

  // Go through layers reading weights
  for (size_t layer = 0; layer < layers; layer++) {
    // read rows and cols
    uint32_t rows;
    uint32_t cols;
    parameters.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    parameters.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    // Read the weights data
    Eigen::MatrixXd weight(rows, cols);
    parameters.read(reinterpret_cast<char*>(weight.data()),
                    rows * cols * sizeof(Eigen::MatrixXd::Scalar));

    std::cout << "Loaded weight: " << weight.rows() << "x" << weight.cols()
              << std::endl;
    ;
    weights.push_back(weight);
  }

  // Go through layers read bias
  for (size_t layer = 0; layer < layers; layer++) {
    // Write rows and cols
    uint32_t rows;
    uint32_t cols;
    parameters.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    parameters.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    // Read the bias data
    Eigen::VectorXd b(rows, cols);
    parameters.read(reinterpret_cast<char*>(b.data()),
                    rows * cols * sizeof(Eigen::VectorXd::Scalar));
    bias.push_back(b);
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
  parameters.close();
}