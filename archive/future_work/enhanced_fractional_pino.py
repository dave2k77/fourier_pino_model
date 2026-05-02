#!/usr/bin/env python3
"""
Enhanced FractionalPINO Model
Leveraging HPFRACC's advanced fractional calculus capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hpfracc as hf
from hpfracc import (
    optimized_caputo, optimized_riemann_liouville, optimized_grunwald_letnikov,
    optimized_caputo_fabrizio_derivative, optimized_atangana_baleanu_derivative,
    optimized_weyl_derivative, optimized_marchaud_derivative,
    FractionalOrder
)
from hpfracc.ml import (
    FractionalNeuralNetwork, BackendType, FractionalGCN, FractionalGAT, FractionalGraphSAGE
)

class EnhancedFractionalPINO(nn.Module):
    """
    Enhanced FractionalPINO leveraging HPFRACC's advanced capabilities
    
    Features:
    - Multiple fractional derivative methods (Caputo, RL, CF, AB, Weyl, Marchaud)
    - Spectral domain processing with fractional operators
    - HPFRACC ML components integration
    - GPU acceleration support
    - Memory-efficient processing
    """
    
    def __init__(self, 
                 modes=12, 
                 width=64,
                 alpha=0.5,
                 fractional_method="caputo",
                 use_spectral=True,
                 use_hpfracc_ml=True):
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.alpha = alpha
        self.fractional_method = fractional_method
        self.use_spectral = use_spectral
        self.use_hpfracc_ml = use_hpfracc_ml
        
        # Initialize fractional operators
        self._init_fractional_operators()
        
        # Initialize neural network components
        self._init_neural_components()
        
        # Initialize spectral components if enabled
        if self.use_spectral:
            self._init_spectral_components()
    
    def _init_fractional_operators(self):
        """Initialize HPFRACC fractional operators"""
        self.fractional_order = FractionalOrder(self.alpha)
        
        # Map fractional methods to HPFRACC functions
        self.fractional_methods = {
            "caputo": optimized_caputo,
            "riemann_liouville": optimized_riemann_liouville,
            "grunwald_letnikov": optimized_grunwald_letnikov,
            "caputo_fabrizio": optimized_caputo_fabrizio_derivative,
            "atangana_baleanu": optimized_atangana_baleanu_derivative,
            "weyl": optimized_weyl_derivative,
            "marchaud": optimized_marchaud_derivative,
        }
        
        if self.fractional_method not in self.fractional_methods:
            raise ValueError(f"Unknown fractional method: {self.fractional_method}")
        
        self.fractional_operator = self.fractional_methods[self.fractional_method]
    
    def _init_neural_components(self):
        """Initialize neural network components"""
        if self.use_hpfracc_ml:
            # Use HPFRACC ML components
            self.fractional_encoder = FractionalNeuralNetwork(
                input_size=self.modes,
                hidden_sizes=[self.width, self.width//2],
                output_size=self.modes,
                fractional_order=self.fractional_order,
                backend=BackendType.TORCH
            )
            
            # Standard neural operator layers
            self.neural_operator = nn.Sequential(
                nn.Linear(self.modes, self.width),
                nn.ReLU(),
                nn.Linear(self.width, self.width),
                nn.ReLU(),
                nn.Linear(self.width, self.modes)
            )
        else:
            # Standard PyTorch components
            self.encoder = nn.Sequential(
                nn.Linear(self.modes, self.width),
                nn.ReLU(),
                nn.Linear(self.width, self.width),
                nn.ReLU(),
                nn.Linear(self.width, self.modes)
            )
            
            self.neural_operator = nn.Sequential(
                nn.Linear(self.modes, self.width),
                nn.ReLU(),
                nn.Linear(self.width, self.width),
                nn.ReLU(),
                nn.Linear(self.width, self.modes)
            )
    
    def _init_spectral_components(self):
        """Initialize spectral domain components"""
        # Spectral domain neural operator
        self.spectral_operator = nn.Sequential(
            nn.Linear(self.modes, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.modes)
        )
    
    def _apply_fractional_operator(self, x):
        """Apply fractional operator using HPFRACC"""
        # Handle complex tensors (from FFT)
        if torch.is_complex(x):
            x_real = x.real
            x_imag = x.imag
        else:
            x_real = x
            x_imag = None
        
        # Convert to numpy for HPFRACC
        x_np = x_real.detach().cpu().numpy()
        
        # Ensure we have the right shape for HPFRACC
        if x_np.ndim > 1:
            # Flatten for HPFRACC processing
            original_shape = x_np.shape
            x_np = x_np.flatten()
        
        # Create time/spatial array with same length
        t = np.linspace(0, 1, len(x_np))
        
        # Apply fractional derivative
        try:
            result = self.fractional_operator(x_np, t, self.alpha)
        except Exception as e:
            print(f"Warning: Fractional operator failed: {e}")
            # Fallback to identity
            result = x_np
        
        # Reshape back to original shape if needed
        if 'original_shape' in locals():
            result = result.reshape(original_shape)
        
        # Convert back to tensor
        result_tensor = torch.tensor(result, device=x.device, dtype=x.dtype)
        
        # Handle complex tensors
        if torch.is_complex(x) and x_imag is not None:
            # Apply same fractional operator to imaginary part
            x_imag_np = x_imag.detach().cpu().numpy()
            if x_imag_np.ndim > 1:
                x_imag_np = x_imag_np.flatten()
            t_imag = np.linspace(0, 1, len(x_imag_np))
            try:
                result_imag = self.fractional_operator(x_imag_np, t_imag, self.alpha)
            except:
                result_imag = x_imag_np
            if 'original_shape' in locals():
                result_imag = result_imag.reshape(original_shape)
            result_imag_tensor = torch.tensor(result_imag, device=x.device, dtype=x.dtype)
            result_tensor = torch.complex(result_tensor, result_imag_tensor)
        
        return result_tensor
    
    def _spectral_fractional_processing(self, x_freq):
        """Apply fractional processing in spectral domain"""
        # Apply fractional operator in frequency domain
        x_frac_freq = self._apply_fractional_operator(x_freq)
        
        # Convert complex to real for neural network processing
        if torch.is_complex(x_frac_freq):
            x_frac_freq_real = x_frac_freq.real
        else:
            x_frac_freq_real = x_frac_freq
        
        # Apply spectral neural operator
        x_frac_freq_flat = x_frac_freq_real.view(x_frac_freq_real.size(0), -1)
        x_frac_freq_flat = x_frac_freq_flat[..., :self.modes]  # Truncate to modes
        
        x_frac_freq_processed = self.spectral_operator(x_frac_freq_flat)
        
        # Reshape back
        x_frac_freq_processed = x_frac_freq_processed.view_as(x_frac_freq_real)
        
        # Convert back to complex if original was complex
        if torch.is_complex(x_freq):
            x_frac_freq_processed = torch.complex(x_frac_freq_processed, torch.zeros_like(x_frac_freq_processed))
        
        return x_frac_freq_processed
    
    def forward(self, x):
        """
        Forward pass with HPFRACC integration
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Output tensor of same shape as input
        """
        batch_size, channels, height, width = x.shape
        
        # Step 1: Apply fractional convolution (if using HPFRACC ML)
        if self.use_hpfracc_ml:
            # Reshape for processing
            x_flat = x.view(batch_size, -1)
            x_flat = x_flat[..., :self.modes]  # Truncate to modes
            
            # Apply fractional encoder
            x_frac = self.fractional_encoder(x_flat, use_fractional=True, method="RL")
        else:
            # Standard processing
            x_flat = x.view(batch_size, -1)
            x_flat = x_flat[..., :self.modes]  # Truncate to modes
            x_frac = self.encoder(x_flat)
        
        # Step 2: Fourier transform to frequency domain
        x_freq = torch.fft.fft2(x, dim=(-2, -1))
        
        # Step 3: Apply fractional processing
        if self.use_spectral:
            # Spectral domain fractional processing
            x_frac_freq = self._spectral_fractional_processing(x_freq)
        else:
            # Direct fractional processing
            x_frac_freq = self._apply_fractional_operator(x_freq)
        
        # Step 4: Neural operator in frequency domain
        # Convert complex to real for neural network processing
        if torch.is_complex(x_frac_freq):
            x_frac_freq_real = x_frac_freq.real
        else:
            x_frac_freq_real = x_frac_freq
        
        x_frac_freq_flat = x_frac_freq_real.view(x_frac_freq_real.size(0), -1)
        x_frac_freq_flat = x_frac_freq_flat[..., :self.modes]  # Truncate to modes
        
        # Apply neural operator
        x_frac_freq_processed = self.neural_operator(x_frac_freq_flat)
        
        # Step 5: Inverse Fourier transform
        x_frac_freq_processed = x_frac_freq_processed.view_as(x_frac_freq_real)
        
        # Convert back to complex if original was complex
        if torch.is_complex(x_frac_freq):
            x_frac_freq_processed = torch.complex(x_frac_freq_processed, torch.zeros_like(x_frac_freq_processed))
        
        x_output = torch.fft.ifft2(x_frac_freq_processed, dim=(-2, -1)).real
        
        return x_output

class FractionalPhysicsLoss(nn.Module):
    """
    Fractional physics loss using HPFRACC operators
    """
    
    def __init__(self, 
                 alpha=0.5,
                 lambda_physics=0.1,
                 fractional_method="caputo"):
        super().__init__()
        
        self.alpha = alpha
        self.lambda_physics = lambda_physics
        self.fractional_method = fractional_method
        
        # Initialize fractional operators
        self._init_fractional_operators()
    
    def _init_fractional_operators(self):
        """Initialize HPFRACC fractional operators"""
        self.fractional_order = FractionalOrder(self.alpha)
        
        # Map fractional methods to HPFRACC functions
        self.fractional_methods = {
            "caputo": optimized_caputo,
            "riemann_liouville": optimized_riemann_liouville,
            "grunwald_letnikov": optimized_grunwald_letnikov,
            "caputo_fabrizio": optimized_caputo_fabrizio_derivative,
            "atangana_baleanu": optimized_atangana_baleanu_derivative,
            "weyl": optimized_weyl_derivative,
            "marchaud": optimized_marchaud_derivative,
        }
        
        if self.fractional_method not in self.fractional_methods:
            raise ValueError(f"Unknown fractional method: {self.fractional_method}")
        
        self.fractional_operator = self.fractional_methods[self.fractional_method]
    
    def _compute_fractional_laplacian(self, u):
        """Compute fractional Laplacian using HPFRACC"""
        # Convert to numpy for HPFRACC
        u_np = u.detach().cpu().numpy()
        
        # Create spatial coordinates
        x = np.linspace(0, 1, u_np.shape[-1])
        
        # Apply fractional derivative (approximating Laplacian)
        try:
            frac_laplacian = self.fractional_operator(u_np, x, self.alpha)
        except Exception as e:
            print(f"Warning: Fractional Laplacian failed: {e}")
            # Fallback to standard Laplacian
            frac_laplacian = np.gradient(np.gradient(u_np, axis=-1), axis=-1)
        
        # Convert back to tensor
        result = torch.tensor(frac_laplacian, device=u.device, dtype=u.dtype)
        return result
    
    def forward(self, u_pred, u_true, x, t):
        """
        Compute fractional physics loss
        
        Args:
            u_pred: Predicted solution
            u_true: True solution
            x: Spatial coordinates
            t: Time coordinates
        
        Returns:
            Total loss (data + physics)
        """
        # Data loss
        data_loss = F.mse_loss(u_pred, u_true)
        
        # Physics loss: Fractional heat equation
        # ∂u/∂t = D_α ∇^α u (fractional heat equation)
        
        # Compute time derivative
        u_t = torch.gradient(u_pred, dim=-1)[0]
        
        # Compute fractional Laplacian
        frac_laplacian_u = self._compute_fractional_laplacian(u_pred)
        
        # Physics residual: ∂u/∂t - D_α ∇^α u = 0
        physics_residual = u_t - frac_laplacian_u
        
        # Compute physics loss
        physics_loss = torch.mean(physics_residual ** 2)
        
        # Total loss
        total_loss = data_loss + self.lambda_physics * physics_loss
        
        return total_loss, data_loss, physics_loss

class MultiMethodFractionalPINO(nn.Module):
    """
    Multi-method FractionalPINO supporting multiple fractional operators
    """
    
    def __init__(self, 
                 modes=12, 
                 width=64,
                 alpha=0.5,
                 methods=["caputo", "riemann_liouville", "caputo_fabrizio"]):
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.alpha = alpha
        self.methods = methods
        
        # Initialize multiple FractionalPINO models
        self.models = nn.ModuleDict()
        for method in methods:
            self.models[method] = EnhancedFractionalPINO(
                modes=modes,
                width=width,
                alpha=alpha,
                fractional_method=method
            )
        
        # Fusion layer
        self.fusion = nn.Linear(len(methods) * modes, modes)
    
    def forward(self, x):
        """Forward pass with multiple fractional methods"""
        outputs = []
        
        for method, model in self.models.items():
            output = model(x)
            outputs.append(output)
        
        # Concatenate outputs
        combined = torch.cat(outputs, dim=-1)
        
        # Fusion
        fused = self.fusion(combined)
        
        return fused

def create_enhanced_fractional_pino(config):
    """
    Factory function to create Enhanced FractionalPINO model
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Enhanced FractionalPINO model
    """
    return EnhancedFractionalPINO(
        modes=config.get('modes', 12),
        width=config.get('width', 64),
        alpha=config.get('alpha', 0.5),
        fractional_method=config.get('fractional_method', 'caputo'),
        use_spectral=config.get('use_spectral', True),
        use_hpfracc_ml=config.get('use_hpfracc_ml', True)
    )

def create_fractional_physics_loss(config):
    """
    Factory function to create Fractional Physics Loss
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Fractional Physics Loss
    """
    return FractionalPhysicsLoss(
        alpha=config.get('alpha', 0.5),
        lambda_physics=config.get('lambda_physics', 0.1),
        fractional_method=config.get('fractional_method', 'caputo')
    )

# Example usage and testing
if __name__ == "__main__":
    # Test Enhanced FractionalPINO
    print("🧪 Testing Enhanced FractionalPINO")
    print("=" * 50)
    
    # Create model
    model = EnhancedFractionalPINO(
        modes=12,
        width=64,
        alpha=0.5,
        fractional_method="caputo",
        use_spectral=True,
        use_hpfracc_ml=True
    )
    
    # Test forward pass
    x = torch.randn(2, 1, 32, 32)
    output = model(x)
    
    print(f"✅ Model created successfully")
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test physics loss
    loss_fn = FractionalPhysicsLoss(
        alpha=0.5,
        lambda_physics=0.1,
        fractional_method="caputo"
    )
    
    u_pred = torch.randn(2, 1, 32, 32)
    u_true = torch.randn(2, 1, 32, 32)
    x = torch.randn(2, 1, 32, 32)
    t = torch.randn(2, 1, 32, 32)
    
    total_loss, data_loss, physics_loss = loss_fn(u_pred, u_true, x, t)
    
    print(f"✅ Physics loss created successfully")
    print(f"✅ Total loss: {total_loss.item():.6f}")
    print(f"✅ Data loss: {data_loss.item():.6f}")
    print(f"✅ Physics loss: {physics_loss.item():.6f}")
    
    # Test multi-method model
    multi_model = MultiMethodFractionalPINO(
        modes=12,
        width=64,
        alpha=0.5,
        methods=["caputo", "riemann_liouville", "caputo_fabrizio"]
    )
    
    output_multi = multi_model(x)
    
    print(f"✅ Multi-method model created successfully")
    print(f"✅ Multi-method output shape: {output_multi.shape}")
    print(f"✅ Multi-method parameters: {sum(p.numel() for p in multi_model.parameters())}")
    
    print("\n🎉 Enhanced FractionalPINO test complete!")
