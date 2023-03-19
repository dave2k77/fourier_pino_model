import numpy as np
from PIL import Image
import os

# specify the directory containing the .npy files
npy_dir = r"C:\Users\davia\OneDrive\Documents\MSc Computer Science\Dissertation\code\images\npz"

# specify the directory to save the .png files
png_dir = r"C:\Users\davia\OneDrive\Documents\MSc Computer Science\Dissertation\code\images\png"

# loop through all .npy files in the directory
for npy_file in os.listdir(npy_dir):
    if npy_file.endswith('.npy'):
        # load the .npy file
        data = np.load(os.path.join(npy_dir, npy_file))
        
        # normalize the data to 0-255 range
        data = (data - np.min(data)) * 255 / (np.max(data) - np.min(data))
        
        # convert the data array to a PIL image
        img = Image.fromarray(data.astype(np.uint8))
        
        # save the PIL image as a .png file
        png_file = os.path.splitext(npy_file)[0] + '.png'
        img.save(os.path.join(png_dir, png_file))