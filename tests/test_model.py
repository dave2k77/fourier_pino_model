#!/usr/bin/env python3
"""
Basic tests for PINO Model

This module contains basic unit tests for the PINO model components.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import torch
import numpy as np

from src.models import PINO_2D_Heat_Equation
from src.layers import FourierTransformLayer, NeuralOperator, InverseFourierTransformLayer


class TestPINOComponents(unittest.TestCase):
    """Test cases for PINO model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.input_size = 32  # Smaller size for faster tests
        
    def test_fourier_transform_layer(self):
        """Test Fourier transform layer."""
        layer = FourierTransformLayer()
        
        # Create test input
        x = torch.randn(self.batch_size, 1, self.input_size, self.input_size)
        
        # Forward pass
        output = layer(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check that output is complex
        self.assertTrue(torch.is_complex(output))
        
    def test_neural_operator(self):
        """Test neural operator layer."""
        layer = NeuralOperator(
            input_size=self.input_size,
            hidden_dims=[64, 128, 64]
        )
        
        # Create test input (complex tensor)
        x = torch.randn(self.batch_size, 1, self.input_size, self.input_size) + \
            1j * torch.randn(self.batch_size, 1, self.input_size, self.input_size)
        
        # Forward pass
        output = layer(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check that output is complex
        self.assertTrue(torch.is_complex(output))
        
    def test_inverse_fourier_transform_layer(self):
        """Test inverse Fourier transform layer."""
        layer = InverseFourierTransformLayer()
        
        # Create test input (complex tensor)
        x = torch.randn(self.batch_size, 1, self.input_size, self.input_size) + \
            1j * torch.randn(self.batch_size, 1, self.input_size, self.input_size)
        
        # Forward pass
        output = layer(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check that output is complex
        self.assertTrue(torch.is_complex(output))
        
    def test_pino_model(self):
        """Test complete PINO model."""
        model = PINO_2D_Heat_Equation(
            input_size=self.input_size,
            hidden_dims=[64, 128, 64]
        )
        
        # Create test input
        x = torch.randn(self.batch_size, 1, self.input_size, self.input_size)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check that output is complex
        self.assertTrue(torch.is_complex(output))
        
    def test_model_info(self):
        """Test model information method."""
        model = PINO_2D_Heat_Equation(
            input_size=self.input_size,
            hidden_dims=[64, 128, 64]
        )
        
        info = model.get_model_info()
        
        # Check required keys
        required_keys = [
            "model_type", "input_size", "hidden_dims", 
            "total_parameters", "trainable_parameters"
        ]
        for key in required_keys:
            self.assertIn(key, info)
            
        # Check parameter counts are positive
        self.assertGreater(info["total_parameters"], 0)
        self.assertGreater(info["trainable_parameters"], 0)
        
    def test_model_parameters(self):
        """Test that model parameters are trainable."""
        model = PINO_2D_Heat_Equation(
            input_size=self.input_size,
            hidden_dims=[64, 128, 64]
        )
        
        # Check that parameters require gradients
        for param in model.parameters():
            self.assertTrue(param.requires_grad)
            
    def test_model_device_transfer(self):
        """Test model transfer to different device."""
        if torch.cuda.is_available():
            model = PINO_2D_Heat_Equation(
                input_size=self.input_size,
                hidden_dims=[64, 128, 64]
            )
            
            # Move to GPU
            model = model.cuda()
            
            # Check that parameters are on GPU
            for param in model.parameters():
                self.assertEqual(param.device.type, "cuda")
                
            # Move back to CPU
            model = model.cpu()
            
            # Check that parameters are on CPU
            for param in model.parameters():
                self.assertEqual(param.device.type, "cpu")


class TestModelGradients(unittest.TestCase):
    """Test cases for model gradients."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 1
        self.input_size = 16
        
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = PINO_2D_Heat_Equation(
            input_size=self.input_size,
            hidden_dims=[32, 64, 32]
        )
        
        # Create test input
        x = torch.randn(self.batch_size, 1, self.input_size, self.input_size, requires_grad=True)
        
        # Forward pass
        output = model(x)
        
        # Compute loss
        target = torch.randn_like(output.real)
        loss = torch.nn.MSELoss()(output.real, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(x.grad)
        
        # Check that model parameters have gradients
        for param in model.parameters():
            self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    unittest.main()
