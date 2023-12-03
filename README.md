# Deep metric learning with Pytorch Lightning

This repository demonstrates a simple implementation of using deep metric learning to create
embeddings for samples present in the `breast cancer` dataset.

The learning process is shown on the gif below:

![learning_process.gif](plots%2Flearning_process.gif)

Majority of results are visible after the first couple of iterations, with clear separation
visible between the classes. Presented results are on the validation set ie. on the unseen datga

## Training setup

The network uses the following parameters to reach the optimum:
 * `learning rate` = 1e-3
 * `epochs` = 25
 * `batch_size` = 32

In regards to specific of the deep metric learning process the network is using:
 * Triplet Margin Loss with 10 triplets created for each batch and margin of 0.3. 
 * Euclidean distance to establish similarity between samples
 * Batch Hard miner

## Bibliography
 * Alexander Hermans, Lucas Beyer, & Bastian Leibe. (2017). In Defense of the Triplet Loss for Person Re-Identification.