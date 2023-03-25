# BALANCING DATA FITTING AND PHYSICAL PROPERTIES: A COMPARATIVE STUDY ON PHYSICS LOSS COEFFICIENTS AND FOURIER ANALYSIS TECHNIQUES IN PINO MODELS FOR PDES

## Introduction

Partial Differential Equations (PDEs) serve as a mathematical bedrock for modelling physical processes, but solving intricate PDEs via traditional numerical 
simulations can be computationally burdensome and limited in capturing inherent randomness. Neural networks present an alternative for addressing physical problems, 
yet often demand extensive training data produced by expensive simulations. Physics-Informed Neural Networks (PINNs) offer a promising surrogate model, merging 
numerical simulation techniques with neural networks and showcasing exceptional accuracy and efficiency in solving parameterised PDEs, even amidst incomplete data 
or ill-posed problems. Nevertheless, PINNs encounter challenges in complex scenarios involving multi-physics, hyperparameter modification, and model generalisation.

This article introduces a cutting-edge approach based on Physics-Informed Neural Operators (PINOs), incorporating Fourier analysis methods into the PINO framework, 
coupled with a bespoke loss function that amalgamates residual loss from the neural operator network and loss from enforcing physical conservation laws governing PDEs, 
as well as Fourier-based derivatives and a physics loss coefficient. Capitalising on the strengths of numerical algorithms to generate training data and visualise 
experimental outcomes, we propose an innovative PINO framework design and provide an exhaustive comparative analysis exploring the influence of varying the physics 
loss coefficient on the performance of PINO models for PDE resolution.


## Model Architecture

The general architecture of our model consists of three major components:

1. an encoder network
2. a neural operator network
3. a decoder network

The diagram below shows the overall structure:



The **_encoder network_** represents neural network that applies the fourier transform on the input data transforming it from the spatial domain to the frequency domain. 
The purpos of doing this is to simplify computational requirements by reducing the complexity of the problem.

The **_neural operator network_** is the core of the PINO model, designed to learn and approximate the underlying PDE operator and the underlying physics of the problem 
by passing the Fourier transformed data it receives from the encoder network and passing it through a series of fully connected layers.

The **_decoder network_** is the final component of the archtecture. it receives the PDE solution predicted by the neural operator network and transforms it back to 
the spatial domain. It does this by applying the Inverse Fourier Transform to the PDE solution.
