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

#ifndef IMAGE_VISUALIZER_HH
#define IMAGE_VISUALIZER_HH

#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

/**
 * Visualizes an image represented as an Eigen::VectorXd by printing it
 * as a grid of pixel values. Assumes a square image (width == height).
 *
 * @param image The image as a flattened Eigen vector (row-major order)
 * @param width The width (and height) of the square image in pixels
 * @param precision Number of decimal places to print (default: 2)
 */
inline void visualize_image(const Eigen::VectorXd& image, int width = 28,
                            int precision = 2) {
  int cols = 0;

  for (auto const& pixel : image) {
    // Format the pixel value and remove trailing zeros
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << pixel;
    std::string pixel_str = oss.str();

    // Remove trailing zeros after decimal point
    if (pixel_str.find('.') != std::string::npos) {
      pixel_str.erase(pixel_str.find_last_not_of('0') + 1, std::string::npos);
      // Remove trailing decimal point if all decimals were zeros
      if (pixel_str.back() == '.') {
        pixel_str.pop_back();
      }
    }

    // Print with fixed width (right-aligned) for even alignment
    std::cout << std::setw(3) << pixel_str << " ";
    cols++;
    if (cols == width) {
      cols = 0;
      std::cout << "\n";
    }
  }
}

#endif  // IMAGE_VISUALIZER_HH
