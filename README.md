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