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
SimpleNN::SimpleNN(std::vector<int> arch)
{
    if(arch.size() < 2)
    {
        throw std::runtime_error("Invalid architecture! it must have at least 2 layers");
    }

    // Create biases and weights
    for(int layer=1; layer < arch.size(); layer++)
    {
        bias.push_back(Eigen::VectorXd(arch[layer]).setRandom());
        // Organize the weights to make the multiplication easier
        // For instance for a 784 30 10 architecture the weights would be
        // 30x784 and 10x30
        // That way when multiplying Wa+b
        // 30x784 * 784x1 ---> 30x1
        // So each row of weights corresponds to the weights of each neuron in the next layer
        weights.push_back(Eigen::MatrixXd(arch[layer], arch[layer-1]).setRandom()); // Organize weights to make the multiplication
    }
    #ifdef DEBUG
    for(auto weight:weights)
    {
        std::cout << "Weights dimensions: " << weight.rows() << " and " << weight.cols() << std::endl;
    }
    for(auto layer_bias:bias)
    {
        std::cout << "Bias dimensions: "<< layer_bias.rows() << " and " << layer_bias.cols() << std::endl;
    }
    #endif
}

Eigen::MatrixXd SimpleNN::forward_propagation(Eigen::VectorXd input)
{
    // Go through the layers and compute the values
    Eigen::MatrixXd activation = input;
    for(int layer = 0; layer < bias.size(); layer++)
    {
        activation = weights[layer]*activation + bias[layer];
        activation = activation_func(activation);
    }

    return activation;
}

Eigen::MatrixXd SimpleNN::activation_func(Eigen::MatrixXd activation)
{
    // Sigmoid function used
    return 1 / (1 + (-activation.array().exp()));
}