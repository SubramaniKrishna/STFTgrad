# Adaptive parameters differentiable STFT

This directory contains code to train a set of STFT window functions to optimize a concentration measure, using gradient descent.

It references [UMNN](https://github.com/AWehenkel/UMNN) to model the mapping from window indices to sample positions.

The code mainly uses PyTorch<sup>1</sup>. For all the other necessary libraries/prerequisites, please create a virtual environment and install them with the command   
~~
pip install -r requirements.txt
~~
The file [changing-parameter.ipynb](./changing-parameter.ipynb) demonstrates the usage of the code in this directory. It can be run to obtain the plots in our paper.

---
### References
[1] https://github.com/pytorch/pytorch

