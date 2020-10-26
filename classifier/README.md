# Classifier with Differentiable STFT front-end

This directory contains code to train a simple classifier with our differentiable STFT front-end to obtain the optimal STFT window length (along with the classifier weights) for minimizing the classification loss. The following figure aims to explain how classification can be affected by choosing an incorrect window size:

![classification_N](./class_snap.png)

The code mainly uses JAX<sup>1</sup> and Haiku<sup>2</sup>. For all the other necessary libraries/prerequisites, please use conda/anaconda to create an environment (from the environment.yml file in this repository) with the command   
~~~
conda env create -f environment.yml
~~~
The file [code_demo.ipynb](./code_demo.ipynb) demonstrates the usage of the code in this directory. It can be run to obtain the plots in our paper.

---
### References
[1] https://github.com/google/jax

[2] https://github.com/deepmind/dm-haiku




