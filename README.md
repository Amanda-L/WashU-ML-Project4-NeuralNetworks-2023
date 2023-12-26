# Project 4: Neural Networks

![image](https://github.com/Amanda-L/WashU-ML-Project4-NeuralNetworks-2023/assets/52643725/8b8eb876-f520-4377-b958-5c3a6a54cced)


This assignment aims to implement a neural network and test it on the Boston dataset, which contains housing prices as targets and community statistics as features. The implementation is organized into three functions (forward pass, compute loss, and back propagation) and a preprocessing step in the file `deepnet.py`. Here is a summarized breakdown of the tasks:

1. **Preprocessing:**
   - Implement the `preprocess` function in `deepnet.py`.
   - Take the training and test data as input.
   - Make the training data zero-mean, and each feature should have standard deviation 1.
   - Use only the training dataset to learn the transformation and apply the same transformations to the test dataset.
   - Transform each input vector by subtracting the mean and dividing by the standard deviation.

2. **Forward Pass:**
   - Implement the `forward_pass` function in `deepnet.py`.
   - Take weights for the network, training data, and the transition function as input.
   - Output the result at each node for the forward pass.
   - W[0] stores the weights for the last layer of the network.

3. **Compute Loss:**
   - Implement the `compute_loss` function in `deepnet.py`.
   - Take the output of the forward pass and the training labels as input.
   - Compute the loss for the entire training set, averaging over all of the points.

4. **Back Propagation (Compute Gradient):**
   - Implement the `backprop` function in `deepnet.py`.
   - Take weights for the network, outputs of the forward pass, training labels, and the derivative of the transition function as input.
   - Use the chain rule to calculate the gradient of the weights.

5. **Visualization (bostondemo.py):**
   - Visualize the RMSE error on the Boston data.
   - Run the demo using `python bostondemo.py`.

The results look like the following graph:
![image](https://github.com/Amanda-L/WashU-ML-Project4-NeuralNetworks-2023/assets/52643725/a8a13357-9856-4418-9d13-da3d867f3f91)

Each dot shows the training/testing error of a house price prediction example. The houses are sorted by increasing price. The very right plot shows the training and testing error.



View Project 4_Neural Networks.pdf and 04Neuralnetworks.html under the Instructions folder for detailed instructions on the assignment.
