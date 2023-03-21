import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

heatmap_folder = "images/heatmaps"
pde_solution_folder = "images/pde_solutions"

class HeatmapPDEDataset(Dataset):
    def __init__(self, heatmap_folder, pde_solution_folder, transform=None):
        self.heatmap_folder = heatmap_folder
        self.pde_solution_folder = pde_solution_folder
        self.transform = transform

        self.heatmap_files = sorted(os.listdir(heatmap_folder))
        self.pde_solution_files = sorted(os.listdir(pde_solution_folder))

    def __len__(self):
        return len(self.heatmap_files)

    def __getitem__(self, idx):
        heatmap_path = os.path.join(self.heatmap_folder, self.heatmap_files[idx])
        pde_solution_path = os.path.join(self.pde_solution_folder, self.pde_solution_files[idx])

        heatmap = np.load(heatmap_path)
        pde_solution = np.load(pde_solution_path)

        if self.transform:
            heatmap = self.transform(heatmap)

        return heatmap, pde_solution

def split_data(dataset, test_size=0.2, random_state=42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_size * dataset_size))

    np.random.seed(random_state)
    np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset

dataset = HeatmapPDEDataset(heatmap_folder, pde_solution_folder, transform=ToTensor())
train_dataset, test_dataset = split_data(dataset)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
