# NeuralNetworkImplementation
Just a simple Feedforward neural network implementation in C++

## What is this?
This is an implementation of a Feedforward neural network in C++ with the opportunity to include AVX instructions (AVX-256) to achieve a better performance.

## Usage
It is not intended to use this code as a library or such things but rather just as a fun project.
I wanted to combine a computational heavy field like deep learning with AVX instructions to take a look at the performance increase.

## Why a look at this might be useful
As I said, this gives a hint of the speedup that is achievable when using the AVX instructions. On my machine with the Intel Core i9 9900k I managed to get a performance increase of around 3x.
