# %%
%load_ext autoreload
%autoreload 2

import sys
import math
import time
import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from sklearn import svm
import pandas as pd
import itertools
from itertools import zip_longest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.optim import Adam, LBFGS

import sys
import os

# Go up 3 levels: from jobs/ → spartan/ → simple-nn/
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.append(project_root)


from model.model_data_class import Cus_Dataset, SimpleNN
from model.train_test_loop import training_loop
from sklearn.preprocessing import StandardScaler


# %%
np.random.seed(1234)
torch.manual_seed(1234)

# %% [markdown]
# Data-Preparation

# %%
# import .csv dataset 
file_path = "../../data/heat_data.csv"
temp1 = pd.read_csv(file_path)

temp2=temp1.copy()

a = temp2.shape[0]

pp1 = np.random.uniform(low=2,high=10,size=a)

temp2['pp1']= pp1
temp2['modtemp'] = temp2['temp']* temp2['pp1']


temp3 = temp2.copy()
cols = ['x', 't', 'pp1','modtemp']
scalers = {}
for col in cols:
    scaler = MinMaxScaler()
    temp3[col] = scaler.fit_transform(temp3[[col]])
    scalers[col] = scaler

# Save the scalers to a file
scaler_file = '../tr-models/scalers.pkl'
with open(scaler_file, 'wb') as f:
    pickle.dump(scalers, f)
    



# %%
# Dataset Preparation

feature_columns = ['x','t','pp1']
target_column = 'modtemp'

train_dataset = Cus_Dataset(temp3,feature_columns,target_column,train_ratio=0.8,\
                                   test_ratio=0.1, val_ratio=0.1,split='train')

val_dataset = Cus_Dataset(temp3,feature_columns,target_column,train_ratio=0.8,\
                                   test_ratio=0.1, val_ratio=0.1,split='val')

test_dataset = Cus_Dataset(temp3,feature_columns,target_column,train_ratio=0.8,\
                                   test_ratio=0.1, val_ratio=0.1,split='test')


# %%
train_dataset.__getitem__(0)

# %%
train_loader = DataLoader(train_dataset, batch_size=512,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=512,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=512,shuffle=True)

print(f"Train dataset size: {len(train_loader)}")

# %%
# check for gpu
if torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

print('Using device:', device)

# %%
input_size = 3
hidden_size = 20
output_size = 1

learning_rate = 0.005
hidden_layers = 5

epochs= 1

model = SimpleNN(input_size,hidden_size,output_size,hidden_layers)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


# %%
loss_train,loss_test,best_model = training_loop(epochs,model,optimizer,train_loader,val_loader)

def move_to_cpu(obj):
    """Recursively move tensors in a dictionary or list to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, list):
        return [move_to_cpu(item) for item in obj]  # Convert tensors inside lists
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}  # Convert tensors inside dicts
    return obj  # Return unchanged for other types

# Ensure all tensors inside lists/dicts are on CPU
loss_train = move_to_cpu(loss_train)
loss_test = move_to_cpu(loss_test)



parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=str, default="000000")
args = parser.parse_args()

# Create a unique folder based on SLURM job ID

# Create a unique folder based on SLURM job ID

folder_path = f"model-files/job_{args.job_id}/"
os.makedirs(folder_path, exist_ok=True)

torch.save(best_model.state_dict(),os.path.join(folder_path, "best-model.pth"))

# Define file path
loss_train_pth = os.path.join(folder_path, "train-loss.pkl")
loss_test_pth = os.path.join(folder_path, "test-loss.pkl")


# Save file in the specified folder
with open(loss_train_pth, "wb") as f:
    pickle.dump(loss_train, f)

with open(loss_test_pth, "wb") as f:
    pickle.dump(loss_test, f)



print(f"File saved at: {loss_train_pth}")
print(f"File saved at: {loss_test_pth}")

print("Training complete")
print(f"Model Architecture: {hidden_size} x {hidden_layers+1}, Learning Rate: {learning_rate},Activation: Tanh, Optimizer: Adam")





# %%
# test the model on the test set






