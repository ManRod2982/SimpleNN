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

#include "image_visualizer.hh"
#include "mnist_reader.hh"
#include "simple_nn.hh"

// Helper function used to get the index of the
// maximum value in a VectorXd
int get_max_index(Eigen::VectorXd vec) {
  auto max_iter = std::max_element(vec.begin(), vec.end());
  return std::distance(vec.begin(), max_iter);
}

int main() {
  // Read MNIST data and create a SimpleNN instance
  auto data = mnist_reader("data/train-images.idx3-ubyte",
                           "data/train-labels.idx1-ubyte");
  // Initialize Neural Network with
  // 784 Input layer (image is 28*28 = 784)
  // 30 Hidden layer
  // 10 Ouput layer
  std::cout << "Creating SimpleNN\n" << std::endl;
  std::vector<int> arch{784, 30, 10};
  SimpleNN nn(arch);
  // Set function to validate data
  // Pass lambda
  nn.set_validation_function(
      [](Eigen::VectorXd output, Eigen::VectorXd label) -> bool {
        // Get the index of the maximum element in the result from the NN
        auto max_it_out = std::max_element(output.begin(), output.end());
        auto output_number = std::distance(output.begin(), max_it_out);

        // Get the index of the max element in the label, to find the label
        auto max_it_lab = std::max_element(label.begin(), label.end());
        auto label_number = std::distance(label.begin(), max_it_lab);

        return label_number == output_number;
      });
  // Set training parameters
  // 30 epochs, 3.0 learning rate, mini batch size of 10 and 10000 images for
  // validation
  nn.set_config(30, 3.0, 10, 10000);

  nn.train(data.images, data.labels);
  std::cout << "Saving parameters" << std::endl;
  nn.save_parameters("training_30_3-0_10");
  auto result = nn.forward_propagation(data.images[0]);
  std::cout << "Result: " << get_max_index(result) << std::endl;
  std::cout << "Target: " << get_max_index(data.labels[0]) << std::endl;

#ifdef DEBUG
  // Visualize the first image to verify it was parsed correctly
  std::cout << "First training image (28x28 MNIST digit):\n";
  visualize_image(data.images[0], 28, 2);
  std::cout << "\n";
#endif

  // Use against validation data
  auto validation = mnist_reader("data/t10k-images.idx3-ubyte",
                                 "data/t10k-labels.idx1-ubyte");
  auto valid = nn.validate_data(validation.images, validation.labels);
  return 0;
}
