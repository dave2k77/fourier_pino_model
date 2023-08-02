import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from PIL import Image

heatmap_folder = r"images\heatmaps"
pde_solution_folder = r"images\pde_solutions"

class HeatmapPDEDataset(Dataset):
    """
    This class takes the paths for the input data and the associated targets and does the following:
    1) resizes the heatmap images and normalise them
    2) transform the resized heatmaps (PDE) in png format to a tensor format
    3) converts the PDE solutions in NumPy's .npz format, reshapes and resizes them 
    4) converts the processed PDE solutions to tensor format. 
    """
    def __init__(self, heatmap_folder, pde_solution_folder, transform=None):
        self.heatmap_folder = heatmap_folder
        self.pde_solution_folder = pde_solution_folder
        self.transform = transform if transform is not None else ToTensor()

        self.heatmap_files = sorted(os.listdir(heatmap_folder))
        self.pde_solution_files = sorted(os.listdir(pde_solution_folder))

    def __len__(self):
        return len(self.heatmap_files)

    def __getitem__(self, idx):
        heatmap_path = os.path.join(self.heatmap_folder, self.heatmap_files[idx])
        pde_solution_path = os.path.join(self.pde_solution_folder, self.pde_solution_files[idx])

        # resize and convert each heatmap image to a greyscale 64 x 64 pixel image.
        heatmap = Image.open(heatmap_path).convert('L').resize((64, 64), Image.ANTIALIAS)

        if self.transform:
            heatmap = self.transform(heatmap)
            heatmap = heatmap.to(torch.float32)

        with np.load(pde_solution_path, allow_pickle=True) as data:
            pde_solution_np = data['u']
            # Resize the pde_solution using scipy.ndimage.zoom
            from scipy.ndimage import zoom
            zoom_factor = (64 / pde_solution_np.shape[0], 64 / pde_solution_np.shape[1])
            pde_solution_np = zoom(pde_solution_np, zoom_factor, order=1)

            pde_solution = torch.tensor(pde_solution_np, dtype=torch.float32)

        return heatmap, pde_solution


def split_data(dataset, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and test datasets.
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_size * dataset_size))

    np.random.seed(random_state)
    np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset

# Create the train and test dataloaders
dataset = HeatmapPDEDataset(heatmap_folder, pde_solution_folder)
train_dataset, test_dataset = split_data(dataset)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
