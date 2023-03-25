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

1. an `encoder network`
2. a `neural operator network`
3. a `decoder network`

The diagram below shows the overall structure:

![Alt text](https://github.com/dave2k77/fourier_pino_model/blob/master/images/PINO%20Architecture%20Diagram.svg)

The `encoder network` represents neural network that applies the fourier transform on the input data transforming it from the spatial domain to the frequency domain. 
The purpos of doing this is to simplify computational requirements by reducing the complexity of the problem.

The `neural operator network` is the core of the PINO model, designed to learn and approximate the underlying PDE operator and the underlying physics of the problem 
by passing the Fourier transformed data it receives from the encoder network and passing it through a series of fully connected layers.

The `decoder network` is the final component of the archtecture. it receives the PDE solution predicted by the neural operator network and transforms it back to 
the spatial domain. It does this by applying the Inverse Fourier Transform to the PDE solution.

The convoluitonal neural network (CNN) was intended to compare the predictions of the PINO model with the ground truth generated by the Finite Difference Method (numerical simulations). However, we chose not implement it opting for a simpler and less sophisticated way of achieving this goal.


## Data Preprocessing

We used a number of helper functions to transform the input data into a format that can easily be used by the PINO model. These helper functions can be found in the `data_preprocess` file found here: [data_preprocess](https://github.com/dave2k77/fourier_pino_model/blob/master/src/dataset_preprocess.py)

The `dataset_preprocess.py` file contains the `HeatmapPDEDataset` class that takes in the location of the input dataset (collection of heatmap png images generated by numerical simulations) and the corresponding PDE solutions (computed from numerical simulations and stored as NumPy's .npz files), and carry out resizing, reshaping and tensor conversion as reuired by the PINO model.

The file also contains the function `split_data()` that takes in the formatted dataset produced by the `HeatmapPDEDataset` class and splits it into training and test datasets which can be used to create dataloaders for the PINO model.


## Building the PINO Model

### The encoder:

We define the encoder neural network class as follows:


    import torch
    import torch.nn as nn
    import numpy as np
    
    class FourierTransformLayer(nn.Module):

        """
        This class is a simple encoder neural network that takes the spatial and temporal coordinates 
        as input and maps them to a higher-dimensional feature space using the 2d fourier transform. 
        """


        def __init__(self):

            super(FourierTransformLayer, self).__init__()


        def forward(self, x):

            # Perform 2D Fourier Transform
            x_fft = torch.fft.fft2(x)

            return x_fft


This `encoder network` represents the `Fourier Transform Layer` that lifts the input data from the spatial domain to the frequency domain. It applies Pytorch's `fft.fft2()`, Pytorch's implementation of the 2D Fast Fourier Transform function, to the input data (Float Tensor object) by applying the `forward()` function to it, returning a `Complex Tensor` object.

This returned `Complex Tensor` object represents the input data in the frequency domain. 

The full code for the Fourier Transform Layer is found here: [encoder](https://github.com/dave2k77/fourier_pino_model/blob/master/src/fourier_transform_layer.py).



### The neural operator network:

This is the main part of the PINO. Its job is to learn the mapping between the PDE space and the latent solution space, and it the process learning the underlying physics of the problem. We implement the neural network operator as follows:
    

    import torch
    import torch.nn as nn

    class NeuralOperator(nn.Module):
        
        
    """
    A neural network with four complex linear layers and GELU activation functions.
    """

    def __init__(self):
    
    
        super(NeuralOperator, self).__init__()

        # GELU activation function
        self.gelu = nn.GELU()

        # First linear layer
        self.fc1 = nn.Linear(64 * 64 * 2, 128)

        # Second linear layer
        self.fc2 = nn.Linear(128, 256)

        # Third linear layer
        self.fc3 = nn.Linear(256, 128)

        # Fourth linear layer and output
        self.fc4 = nn.Linear(128, 64 * 64 * 2)

    def forward(self, x):
        # Convert complex tensor to real-valued tensor with two channels: real and imaginary parts
        x = torch.view_as_real(x)

        # Flatten the input tensor
        x = torch.flatten(x, start_dim=1)  # Flatten the tensor, keeping the batch dimension

        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))
        x = self.fc4(x)

        # Reshape the output tensor back to the original shape with two channels (real and imaginary parts)
        x = x.view(-1, 64, 64, 2)

        # Convert the real-valued tensor back to a complex tensor
        x = torch.view_as_complex(x)

        return x


The neural operator network consists of a series of fully connected hidden layers (`self.fc()`) activated by the GELU (Guassian Error Linear Units) activation (`self.gelu()`) function. The data that comes in is a Complex Tensor (has real and imaginary parts), however, the GELU activation is defined only for real-valued inputs so the have to convert the data to a reaf-valued tensor (Float Tensor) before applying the GELU activation. 

We then need to reshape and reconvert the real-valued tensor back to a complex tenesor before returning the ouput.

The full code for the Neural Operator Layer can be found here: [neural_operator]()
