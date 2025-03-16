# Perceptron 

A simple Perceptron implementation in C++ that reads data from a CSV file, trains on it, and makes predictions.

## Features 

- Reads dataset from a CSV file.
- Implements a basic Perceptron learning algorithm.
- Trains on labeled data and makes predictions.
- Can Do Linear Regression 

## File Structure

- `b_data.csv` - Dataset used for training/testing.
- `loadcsv.h` / `loadcsv.cpp` - Functions for loading CSV data.
- `perceptron.h` - Implementation of the Perceptron model.
- `main.cpp` - Runs the Perceptron model with the dataset.

## Compilation & Usage

```sh
g++ main.cpp loadcsv.cpp -o main
./main
