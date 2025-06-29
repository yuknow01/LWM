import pickle
import scipy.io
import torch
import numpy as np

# Load the .p file
with open('filename.p', 'rb') as file:
    tensor_data = pickle.load(file)  # This should be a PyTorch tensor

# Ensure it is a PyTorch tensor
if isinstance(tensor_data, torch.Tensor):
    tensor_data = tensor_data.numpy()  # Convert to NumPy array

# Save as a MATLAB file
scipy.io.savemat('filename.mat', {'tensor': tensor_data})

#%% In MATLAB

# data = load('filename.mat');
# tensor = data.tensor;  % Access the variable
