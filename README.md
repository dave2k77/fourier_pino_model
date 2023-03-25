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


## Creating the Dataset

Consider the following scenario where we have a rectangular region with a hot edge (100 degrees celsius) and all other edges remaining cold at 0 degrees celsius.

![Alt text](https://github.com/dave2k77/fourier_pino_model/blob/master/images/HeatTransferDomain.svg)

In order to analysis the temperature change of the rectangular region as time evolves, we need to access or sample various points in the rectangular region. To do this, we can discretise the rectangular domain using the following scheme:

![Alt text](https://github.com/dave2k77/fourier_pino_model/blob/master/images/DiscretisationStrategy.svg)

where each (x, y) coordinate represents a grid point in the domain.

We then used the Finite Difference Approximation Scheme (show below), we can compute the temperature at finitely many points in the rectangular domain and use it to paint a picture of how the heat energy diffuses throughout the domain as time evolves. 

![image](https://user-images.githubusercontent.com/30156495/227734579-3670a692-49b9-4b76-bb1c-6de515ec3af5.png)


The result of this process for different initial conditions are shown below:

![](https://github.com/dave2k77/fourier_pino_model/blob/master/movies/heat_equation_solution_alpha10.gif)

Solution of the 2D Heat Equations with alpha = 1 and initial temperature distribution u(x, y, 0) = 0

![](https://github.com/dave2k77/fourier_pino_model/blob/master/movies/heat_equation_solution_alpha10_u25.gif)

Solution of the 2D Heat Equations with alpha = 1 and initial temperature distribution u(x, y, 0) = 25

![](https://github.com/dave2k77/fourier_pino_model/blob/master/movies/heat_equation_solution_alpha10_u50.gif)

Solution of the 2D Heat Equations with alpha = 1 and initial temperature distribution u(x, y, 0) = 50


To create our dataset, we sampled the evolution for different initial temperature distributions storing each snapshot as a heatmap in png format. To create the target dataset, we saved the corresponding solutions (to each heatmap) as a .npz file. The code for generating the dataset can be found here: [fdm_data_generator](https://github.com/dave2k77/fourier_pino_model/blob/master/src/fdm_2d_heat_eqn.py)



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



The neural operator network consists of a series of fully connected hidden layers (`self.fc()`) activated by the GELU (Guassian Error Linear Units) activation (`self.gelu()`) function. 

The data that comes in is a complex tensor (has real and imaginary parts), however, the GELU activation is defined only for real-valued inputs so the have to convert the data to a reaf-valued tensor (Float Tensor) before applying the GELU activation. 

We then need to reshape and reconvert the real-valued tensor back to a complex tenesor before returning the ouput. The full code for the Neural Operator Layer can be found here: [neural_operator](https://github.com/dave2k77/fourier_pino_model/blob/master/src/neural_operator_layer.py)


### The decoder network:

The `decoder network` represents the Inverse Transform Layer that returns the output of the neural operator (predicted solution) from the frequency domain to the spatial domain. It s does this by applying Pytorch's implementation of the Inverse Fourier Transform, `fft.ifft2()`, to the neural network's output. We implemented this layer as follows:


    import torch
    import torch.nn as nn

    class InverseFourierTransformLayer(nn.Module):
        """
        This class implements a simple decoder neural network that returns 
        the output from operator network from the frequency domain back to the 
        the spatial domain using the 2d inverse fourier transform.
        """

        def __init__(self):

            super(InverseFourierTransformLayer, self).__init__()


        def forward(self, x):

            # Perform 2D Inverse Fourier Transform
            x_ifft = torch.fft.ifft2(x)

            return x_ifft
            

The output from theis layer represents the predicted solution from the PINO. The full code for the InverseTransformLayer is found here: [decoder](https://github.com/dave2k77/fourier_pino_model/blob/master/src/inverse_transform_layer.py)


## Building the PINO

Now that we have our dataset formatted and ready, and all the layes of our PINO model are built, it is now time to put everything together and assemble our PINO model. The implementation is shown below:


    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from dataset_builder import HeatmapPDEDataset, split_data
    from preprocess import *
    from fourier_transform_layer import FourierTransformLayer
    from inverse_transform_layer import InverseFourierTransformLayer
    from neural_operator_layer import NeuralOperator


    class PINO_2D_Heat_Equation(nn.Module):
        def __init__(self):
            super(PINO_2D_Heat_Equation, self).__init__()
            self.encoder = FourierTransformLayer()
            self.neural_operator = NeuralOperator()
            self.decoder = InverseFourierTransformLayer()

        def forward(self, x):
            x = self.encoder(x)  # returns x: complex float
            x = self.neural_operator(x)
            x = self.decoder(x)
            return x


Notice that the `__init__` function initialises all the layers of the PINO. Once data is pass to the PINO model, the data will be pushed thorugh each layer by the `forward` function before returning the output. This output is the predicted solution to the PDE.


## Training the PINO Model

To carry out the training process, we implemented a number of utility functions to help with data plotting, defining the loss function,  calculating accuracy scores, and implementing the training function. The full code can be found here: [utility](https://github.com/dave2k77/fourier_pino_model/blob/master/src/utility_functions.py)

To training the PINO, we adapt the following hyperparameter strategy:

![](https://github.com/dave2k77/fourier_pino_model/blob/master/images/Hyperparameters.svg)

To train the model, we do the following:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Replace these paths with the correct ones for your dataset
    heatmap_folder = r"images\heatmaps"
    pde_solution_folder = r"images\pde_solutions"

    # Initilise PINO 2D Heat Equation Model
    model = PINO_2D_Heat_Equation()
    model.to(device)

    # Load training and test data
    data = HeatmapPDEDataset(heatmap_folder, pde_solution_folder)
    train_dataset, test_dataset = split_data(data)
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    # Set hyperparameters
    num_epochs = 100  # options: 100, 250, 500

    # physics loss coefficients
    physics_loss_coefficient_A = 0.001
    physics_loss_coefficient_B = 0.01 
    physics_loss_coefficient_C = 0.1

    # Learning Rates
    lr_1 = 0.001
    lr_2 = 0.005
    lr_3 = 0.01

    # Experiment A Optimizers
    optimizerA_1 = optim.SGD(model.parameters(), lr=lr_1)
    optimizerA_2 = optim.SGD(model.parameters(), lr=lr_2)
    optimizerA_3 = optim.SGD(model.parameters(), lr=lr_3)

    # Experiment B Optimzers
    optimizerB_1 = optim.Adam(model.parameters(), lr=lr_1)
    optimizerB_2 = optim.Adam(model.parameters(), lr=lr_2)
    optimizerB_3 = optim.Adam(model.parameters(), lr=lr_3)
    
    # loss function
    loss_fn = loss_function

    # train the model
    train_loss_history, test_loss_history, mean_r2_score = train(model=model, loss_fn=loss_fn, optimizer=optimizerB_2,
                                                  train_loader=train_loader, test_loader=test_loader,
                                                  num_epochs=num_epochs, physics_loss_coefficient=physics_loss_coefficient_B)
    plot_loss(train_loss_history, test_loss_history, save=True, save_path=r'graphs\train_test-phyloss-01.png')


The details of the `train` function and the `loss_function` can be found in the [utility]() file.


## Results

