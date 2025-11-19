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
#include "mnist_reader.hh"

#include <fstream>
#include <iostream>

// MNIST Magic number for training images
constexpr uint32_t mnist_images_hdr = 0x00000803;
constexpr uint32_t mnist_labels_hdr = 0x00000801;

// Helper function to switch the endinaness
uint32_t swap_endian(uint32_t val) {
  return (val << 24) | ((val << 8) & 0x00FF0000) | ((val >> 8) & 0x0000FF00) |
         (val >> 24);
}

// Function to parse the image data into a matrix
// it vectorizes the image into a single vector and returns as many images, for
// instance if there are 50 images of 2x2 it returns a vector of vectors
// containing 50x4x1
std::vector<std::vector<uint8_t>> parse_images(const std::string &images_path) {
  // Implementation for parsing MNIST images
  std::cout << "Parsing images from: " << images_path << std::endl;
  try {
    std::ifstream file(images_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::ifstream::failure("Could not open file");
    }
    // Check magic number
    uint32_t magic_number = 0;
    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    magic_number = swap_endian(magic_number);
    std::cout << "Magic Number: " << std::hex << magic_number << std::dec
              << std::endl;
    if (magic_number != mnist_images_hdr) {
      throw std::ifstream::failure("Invalid MNIST image file");
    }
    // Read number of images, rows, and columns
    uint32_t num_images = 0;
    file.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
    num_images = swap_endian(num_images);
    std::cout << "Number of Images: " << num_images << std::endl;
    uint32_t num_rows = 0;
    file.read(reinterpret_cast<char *>(&num_rows), sizeof(num_rows));
    num_rows = swap_endian(num_rows);
    uint32_t num_cols = 0;
    file.read(reinterpret_cast<char *>(&num_cols), sizeof(num_cols));
    num_cols = swap_endian(num_cols);
    std::cout << "Image Size: " << num_rows << "x" << num_cols << std::endl;

    // Read image data
    std::vector<std::vector<uint8_t>> images(
        num_images, std::vector<uint8_t>(num_rows * num_cols, 0));
    for (uint32_t i = 0; i < num_images; ++i) {
      // Pixels are organized row wise
      // | 0 1 2 |
      // | 3 4 5 |
      // | 6 7 8 |
      // translated to [0 1 2 3 4 5 6 7 8]
      // We need to pack the image into a vector
      // | p00 p01 |
      // | p10 p11 |
      // | p20 p21 |
      // Example if the image was 3x2 that's how the pixels would be in a
      // vector [p00 p01 p10 p11 p20 p21] [ 0   1   2   3   4   5]
      std::vector<uint8_t> image(num_rows * num_cols);
      file.read(reinterpret_cast<char *>(image.data()), image.size() * sizeof(uint8_t));

      images[i] = image;
    }

    return images;

  } catch (const std::ifstream::failure &e) {
    std::cerr << "Exception opening/reading file: " << e.what() << std::endl;
  }

  return {{}};
}

// Function used to parse label data into a vector
std::vector<uint8_t> parse_labels(const std::string &labels_path) {
  // Implementation for parsing MNIST labels
  std::cout << "Parsing labels from: " << labels_path << std::endl;
  try {
    std::ifstream file(labels_path, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Unable to open the file: " << labels_path << std::endl;
      return {};
    }

    // Check Magic number
    uint32_t magic_number = 0U;
    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    magic_number = swap_endian(magic_number);
    if (magic_number != mnist_labels_hdr) {
      std::cerr << "Invalid data header! Magic number: " << magic_number
                << std::endl;
      return {};
    }

    // Get number of labels
    uint32_t labels_num = 0U;
    file.read(reinterpret_cast<char *>(&labels_num), sizeof(labels_num));
    labels_num = swap_endian(labels_num);
    std::cout << "Number of labels: " << labels_num << std::endl;

    // Get all labels and store them in a vector TODO change to an eigen type
    std::vector<uint8_t> labels = std::vector<uint8_t>(labels_num, 10);
    for (size_t i = 0; i < labels_num; i++) {
      file.read(reinterpret_cast<char *>(&(labels[i])), sizeof(labels[i]));
    }

    return labels;
  } catch (const std::exception &e) {
    std::cerr << "Exception opening/reading file: " << e.what() << std::endl;
  }

  return {};
}

// Implementation for MNISTReader
// It returns a matrix with images and a corresponding vector with labels
// it requires the images path and the labels path
std::tuple<std::vector<std::vector<uint8_t>>, std::vector<uint8_t>>
mnist_reader(const std::string &images_path, const std::string &labels_path) {
  auto images = parse_images(images_path);
  auto labels = parse_labels(labels_path);

  return std::make_tuple(images, labels);
}
