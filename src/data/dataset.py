"""
Dataset Classes for PINO Model

This module provides dataset classes for loading and preprocessing heat equation data
for training the PINO model.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
from typing import Tuple, Optional, List
import glob


class HeatmapPDEDataset(Dataset):
    """
    Dataset class for loading heatmap images and corresponding PDE solutions.
    
    This dataset loads heatmap images (PNG format) and their corresponding
    PDE solutions (NPZ format) for training the PINO model.
    """
    
    def __init__(self, heatmap_folder: str, pde_solution_folder: str, 
                 transform_size: Tuple[int, int] = (64, 64)):
        """
        Initialize the dataset.
        
        Args:
            heatmap_folder: Path to folder containing heatmap images
            pde_solution_folder: Path to folder containing PDE solution files
            transform_size: Target size for resizing images (height, width)
            
        Raises:
            FileNotFoundError: If folders don't exist
            ValueError: If no matching files found
        """
        self.heatmap_folder = heatmap_folder
        self.pde_solution_folder = pde_solution_folder
        self.transform_size = transform_size
        
        # Validate folders exist
        if not os.path.exists(heatmap_folder):
            raise FileNotFoundError(f"Heatmap folder not found: {heatmap_folder}")
        if not os.path.exists(pde_solution_folder):
            raise FileNotFoundError(f"PDE solution folder not found: {pde_solution_folder}")
        
        # Get file lists
        self.heatmap_files = sorted(glob.glob(os.path.join(heatmap_folder, "*.png")))
        self.pde_files = sorted(glob.glob(os.path.join(pde_solution_folder, "*.npz")))
        
        if not self.heatmap_files:
            raise ValueError(f"No PNG files found in {heatmap_folder}")
        if not self.pde_files:
            raise ValueError(f"No NPZ files found in {pde_solution_folder}")
        
        # Validate file counts match
        if len(self.heatmap_files) != len(self.pde_files):
            raise ValueError(
                f"Number of heatmap files ({len(self.heatmap_files)}) "
                f"does not match number of PDE files ({len(self.pde_files)})"
            )
        
        print(f"Loaded {len(self.heatmap_files)} data pairs")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.heatmap_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (heatmap_tensor, pde_solution_tensor)
            
        Raises:
            IndexError: If idx is out of range
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Load heatmap image
        heatmap_path = self.heatmap_files[idx]
        heatmap_image = Image.open(heatmap_path).convert('L')  # Convert to grayscale
        heatmap_image = heatmap_image.resize(self.transform_size, Image.LANCZOS)
        heatmap_tensor = torch.from_numpy(np.array(heatmap_image)).float() / 255.0
        
        # Add channel dimension
        heatmap_tensor = heatmap_tensor.unsqueeze(0)  # (1, H, W)
        
        # Load PDE solution
        pde_path = self.pde_files[idx]
        pde_data = np.load(pde_path)
        pde_solution = pde_data['solution'] if 'solution' in pde_data else pde_data['arr_0']
        pde_tensor = torch.from_numpy(pde_solution).float()
        
        # Ensure PDE solution has correct shape
        if pde_tensor.dim() == 2:
            pde_tensor = pde_tensor.unsqueeze(0)  # Add channel dimension
        
        return heatmap_tensor, pde_tensor
    
    def get_dataset_info(self) -> dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            "num_samples": len(self),
            "heatmap_folder": self.heatmap_folder,
            "pde_solution_folder": self.pde_solution_folder,
            "transform_size": self.transform_size,
            "heatmap_files": len(self.heatmap_files),
            "pde_files": len(self.pde_files)
        }


def split_data(dataset: HeatmapPDEDataset, 
               train_ratio: float = 0.8, 
               random_seed: int = 42) -> Tuple[HeatmapPDEDataset, HeatmapPDEDataset]:
    """
    Split dataset into training and test sets.
    
    Args:
        dataset: The dataset to split
        train_ratio: Ratio of training data (default: 0.8)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, test_dataset)
        
    Raises:
        ValueError: If train_ratio is not between 0 and 1
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    
    # Split the dataset
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )
    
    print(f"Split dataset: {train_size} training samples, {test_size} test samples")
    
    return train_dataset, test_dataset
