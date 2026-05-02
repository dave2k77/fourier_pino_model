#!/usr/bin/env python3
"""
Fractional Data Generator for FractionalPINO Experiments (Fixed Version)
Generate synthetic data for fractional PDE problems with correct tensor shapes
"""

import torch
import numpy as np
from typing import Tuple, Optional

def generate_fractional_heat_data(
    alpha: float = 0.5,
    domain_size: int = 32,
    time_steps: int = 100,
    batch_size: int = 100,
    noise_level: float = 0.0
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate synthetic data for fractional heat equation
    
    Problem: ∂u/∂t = D_α ∇^α u
    Analytical solution: u(x,y,t) = sin(πx)sin(πy)exp(-D_α(π²)^α t)
    
    Args:
        alpha: Fractional order
        domain_size: Spatial domain size (N×N)
        time_steps: Number of time steps
        batch_size: Number of samples
        noise_level: Noise level for data
    
    Returns:
        Tuple of (train_data, test_data) where each is (x, y)
    """
    
    # Create spatial grid
    x = torch.linspace(0, 1, domain_size)
    y = torch.linspace(0, 1, domain_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create time grid
    t = torch.linspace(0, 1, time_steps)
    
    # Parameters
    D_alpha = 1.0  # Diffusion coefficient
    
    # Generate analytical solution
    u_analytical = torch.zeros(batch_size, time_steps, domain_size, domain_size)
    
    for i in range(batch_size):
        for j, t_val in enumerate(t):
            # Analytical solution: u(x,y,t) = sin(πx)sin(πy)exp(-D_α(π²)^α t)
            u_analytical[i, j] = (torch.sin(np.pi * X) * torch.sin(np.pi * Y) * 
                                torch.exp(-D_alpha * (np.pi**2)**alpha * t_val))
    
    # Add noise if specified
    if noise_level > 0:
        noise = torch.randn_like(u_analytical) * noise_level
        u_analytical += noise
    
    # Create input-output pairs for neural operator
    # Input: spatial field at time t
    # Output: spatial field at time t+1
    
    # Create input tensor: [batch_size, 1, height, width] (spatial field)
    # Create output tensor: [batch_size, 1, height, width] (spatial field)
    
    inputs = torch.zeros(batch_size, 1, domain_size, domain_size)
    outputs = torch.zeros(batch_size, 1, domain_size, domain_size)
    
    for i in range(batch_size):
        # Use first time step as input
        inputs[i, 0] = u_analytical[i, 0]
        # Use second time step as output (or last time step if only one)
        if time_steps > 1:
            outputs[i, 0] = u_analytical[i, 1]
        else:
            outputs[i, 0] = u_analytical[i, 0]
    
    # Split into train/test
    train_size = int(0.8 * batch_size)
    
    train_inputs = inputs[:train_size]
    train_outputs = outputs[:train_size]
    test_inputs = inputs[train_size:]
    test_outputs = outputs[train_size:]
    
    return (train_inputs, train_outputs), (test_inputs, test_outputs)

def generate_fractional_wave_data(
    alpha: float = 0.5,
    domain_size: int = 32,
    time_steps: int = 100,
    batch_size: int = 100,
    noise_level: float = 0.0
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate synthetic data for fractional wave equation
    
    Problem: ∂²u/∂t² = c² ∇^α u
    Analytical solution: u(x,y,t) = sin(πx)sin(πy)cos(c(π²)^(α/2) t)
    
    Args:
        alpha: Fractional order
        domain_size: Spatial domain size (N×N)
        time_steps: Number of time steps
        batch_size: Number of samples
        noise_level: Noise level for data
    
    Returns:
        Tuple of (train_data, test_data) where each is (x, y)
    """
    
    # Create spatial grid
    x = torch.linspace(0, 1, domain_size)
    y = torch.linspace(0, 1, domain_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create time grid
    t = torch.linspace(0, 1, time_steps)
    
    # Parameters
    c = 1.0  # Wave speed
    
    # Generate analytical solution
    u_analytical = torch.zeros(batch_size, time_steps, domain_size, domain_size)
    
    for i in range(batch_size):
        for j, t_val in enumerate(t):
            # Analytical solution: u(x,y,t) = sin(πx)sin(πy)cos(c(π²)^(α/2) t)
            u_analytical[i, j] = (torch.sin(np.pi * X) * torch.sin(np.pi * Y) * 
                                torch.cos(c * (np.pi**2)**(alpha/2) * t_val))
    
    # Add noise if specified
    if noise_level > 0:
        noise = torch.randn_like(u_analytical) * noise_level
        u_analytical += noise
    
    # Create input-output pairs for neural operator
    inputs = torch.zeros(batch_size, 1, domain_size, domain_size)
    outputs = torch.zeros(batch_size, 1, domain_size, domain_size)
    
    for i in range(batch_size):
        # Use first time step as input
        inputs[i, 0] = u_analytical[i, 0]
        # Use second time step as output (or last time step if only one)
        if time_steps > 1:
            outputs[i, 0] = u_analytical[i, 1]
        else:
            outputs[i, 0] = u_analytical[i, 0]
    
    # Split into train/test
    train_size = int(0.8 * batch_size)
    
    train_inputs = inputs[:train_size]
    train_outputs = outputs[:train_size]
    test_inputs = inputs[train_size:]
    test_outputs = outputs[train_size:]
    
    return (train_inputs, train_outputs), (test_inputs, test_outputs)

def generate_fractional_diffusion_data(
    alpha: float = 0.5,
    domain_size: int = 32,
    time_steps: int = 100,
    batch_size: int = 100,
    noise_level: float = 0.0
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate synthetic data for fractional diffusion equation with source term
    
    Problem: ∂u/∂t = D_α ∇^α u + f(x,y,t)
    Source term: f(x,y,t) = sin(πx)sin(πy)exp(-t)
    Analytical solution: u(x,y,t) = sin(πx)sin(πy)exp(-t)
    
    Args:
        alpha: Fractional order
        domain_size: Spatial domain size (N×N)
        time_steps: Number of time steps
        batch_size: Number of samples
        noise_level: Noise level for data
    
    Returns:
        Tuple of (train_data, test_data) where each is (x, y)
    """
    
    # Create spatial grid
    x = torch.linspace(0, 1, domain_size)
    y = torch.linspace(0, 1, domain_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create time grid
    t = torch.linspace(0, 1, time_steps)
    
    # Generate analytical solution
    u_analytical = torch.zeros(batch_size, time_steps, domain_size, domain_size)
    
    for i in range(batch_size):
        for j, t_val in enumerate(t):
            # Analytical solution: u(x,y,t) = sin(πx)sin(πy)exp(-t)
            u_analytical[i, j] = (torch.sin(np.pi * X) * torch.sin(np.pi * Y) * 
                                torch.exp(-t_val))
    
    # Add noise if specified
    if noise_level > 0:
        noise = torch.randn_like(u_analytical) * noise_level
        u_analytical += noise
    
    # Create input-output pairs for neural operator
    inputs = torch.zeros(batch_size, 1, domain_size, domain_size)
    outputs = torch.zeros(batch_size, 1, domain_size, domain_size)
    
    for i in range(batch_size):
        # Use first time step as input
        inputs[i, 0] = u_analytical[i, 0]
        # Use second time step as output (or last time step if only one)
        if time_steps > 1:
            outputs[i, 0] = u_analytical[i, 1]
        else:
            outputs[i, 0] = u_analytical[i, 0]
    
    # Split into train/test
    train_size = int(0.8 * batch_size)
    
    train_inputs = inputs[:train_size]
    train_outputs = outputs[:train_size]
    test_inputs = inputs[train_size:]
    test_outputs = outputs[train_size:]
    
    return (train_inputs, train_outputs), (test_inputs, test_outputs)

def generate_multi_scale_data(
    alpha: float = 0.5,
    domain_size: int = 32,
    time_steps: int = 100,
    batch_size: int = 100,
    scales: list = [1, 10, 100],
    noise_level: float = 0.0
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate synthetic data for multi-scale fractional PDE
    
    Problem: Multi-scale fractional PDE with varying scales
    Analytical solution: u(x,y,t) = Σ_scale sin(scale*πx)sin(scale*πy)exp(-scale*t)
    
    Args:
        alpha: Fractional order
        domain_size: Spatial domain size (N×N)
        time_steps: Number of time steps
        batch_size: Number of samples
        scales: List of scales
        noise_level: Noise level for data
    
    Returns:
        Tuple of (train_data, test_data) where each is (x, y)
    """
    
    # Create spatial grid
    x = torch.linspace(0, 1, domain_size)
    y = torch.linspace(0, 1, domain_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create time grid
    t = torch.linspace(0, 1, time_steps)
    
    # Generate analytical solution
    u_analytical = torch.zeros(batch_size, time_steps, domain_size, domain_size)
    
    for i in range(batch_size):
        for j, t_val in enumerate(t):
            solution = torch.zeros_like(X)
            for scale in scales:
                solution += (torch.sin(scale * np.pi * X) * torch.sin(scale * np.pi * Y) * 
                           torch.exp(-scale * t_val))
            u_analytical[i, j] = solution
    
    # Add noise if specified
    if noise_level > 0:
        noise = torch.randn_like(u_analytical) * noise_level
        u_analytical += noise
    
    # Create input-output pairs for neural operator
    inputs = torch.zeros(batch_size, 1, domain_size, domain_size)
    outputs = torch.zeros(batch_size, 1, domain_size, domain_size)
    
    for i in range(batch_size):
        # Use first time step as input
        inputs[i, 0] = u_analytical[i, 0]
        # Use second time step as output (or last time step if only one)
        if time_steps > 1:
            outputs[i, 0] = u_analytical[i, 1]
        else:
            outputs[i, 0] = u_analytical[i, 0]
    
    # Split into train/test
    train_size = int(0.8 * batch_size)
    
    train_inputs = inputs[:train_size]
    train_outputs = outputs[:train_size]
    test_inputs = inputs[train_size:]
    test_outputs = outputs[train_size:]
    
    return (train_inputs, train_outputs), (test_inputs, test_outputs)
