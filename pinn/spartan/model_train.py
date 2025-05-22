import sys
import json
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

current_dir = os.path.dirname(os.path.abspath(__file__))
training_data_dir = os.path.join(current_dir, '../training_data')
model_dir = os.path.join(current_dir, '../')
sys.path.insert(0,str(training_data_dir))
sys.path.insert(0,str(model_dir))

 
    

from training_data.simdata_mush_dirc import  fdd, pdeinp, icinp, bcinp,HT_sim ,scaler, invscaler
from Model.loss_func_sol import loss_fn_data,pde_loss,ic_loss,boundary_loss
from Model.train_testloop import *


settings_path_1 = os.path.join(current_dir, '..', 'training_data', 'settings.json')
with open(settings_path_1,'r') as file:
    settings = json.load(file)
   


heat_data = HT_sim(settings)
alpha = heat_data.alpha_l
tempfield = heat_data.datagen()

# heat_data.plot_temp(25)
dt = heat_data.dt
dx = heat_data.dx
# print(heat_data.dx)
# print(dt)

settings_path = os.path.join(current_dir, '..', 'training_data', 'settings.json')
with open(settings_path,'r') as file:
    props = json.load(file)
    

temp_data = tempfield.flatten()

# def temp_scaler(temp_data, temp_init, t_surr):
#     temp_data = (temp_data - t_surr) / (temp_init - t_surr)
#     return temp_data

# # temp_data = scaler(temp_data,400.0,919.0)

# temp_data_s = temp_scaler(temp_data, temp_init, t_surr)


num_steps = tempfield.shape[0]
numpoints = tempfield.shape[1] 

pde_pts= props['pde_pts']
ic_pts = props['ic_pts']
bc_pts = props['bc_pts']

# x_c = 1/length
# t_c = (alpha/(length**2))
# temp_c = 919.0
time_end = props['time_end'] # 0.5
length = props['length'] # 0.015

inp_data = fdd(15e-3, time_end, numpoints, num_steps)


# def scale2(x,x_c,t_c):
#     scaled_x = x.copy()
#     scaled_x[:,0] = x[:,0] * x_c
#     scaled_x[:,1] = x[:,1] * t_c
#     return scaled_x

# inp_data2 = scale2(inp_data,x_c,t_c)

# input dataset-pde residual
# The pde inputs are generated using the pdeinp function in simdata.py
pde_data = pdeinp(dx,length-dx,dt,time_end,pde_pts,"Sobol",scl="False") 

# pde_data2 = scale2(pde_data,x_c,t_c)

# input dataset - ic residual
ic_data = icinp(length,ic_pts,scl="False")
# ic_data2 = scale2(ic_data,x_c,t_c)
# input dataset - boundary residual
bc_ldata = bcinp(length,time_end,bc_pts,dt,scl="False")[0]
bc_rdata = bcinp(length,time_end,bc_pts,dt,scl="False")[1]

# bc_ldata2 = scale2(bc_ldata,x_c,t_c)
# bc_rdata2 = scale2(bc_rdata,x_c,t_c)

if torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

# print('Using device:', device)

# %% [markdown]
# ### Tensor inputs

input_t = torch.tensor(inp_data).float().to(device)
inp_pdet = torch.tensor(pde_data).float().to(device)
inp_ict = torch.tensor(ic_data).float().to(device)
inp_bclt = torch.tensor(bc_ldata).float().to(device)
inp_bclr = torch.tensor(bc_rdata).float().to(device)


# Tensor the target values
temp_t = torch.tensor(temp_data).float().to(device)
temp_t = temp_t.view(-1,1)

# temp_init_s= temp_scaler(919.0, temp_init, t_surr)
# temp_init = scaler(temp_init,500.0,919.0)
temp_init = props['temp_init']                   #  K -Initial Temperature (919 c) AL 380
temp_init_t = torch.tensor(temp_init).float().to(device)
T_L = props['T_L']                   #  K -Liquidus Temperature (615 c) AL 380
# T_L_s = scaler(T_L,temp_init, t_surr)                     #  K -Liquidus Temperature (615 c) AL 380
# T_L = scaler(T_L,500.0,919.0)
T_S = props['T_S']                   #  K -Solidus Temperature (615 c) AL 380
# T_S_s = scaler(T_S,temp_init, t_surr)                     #  K -Solidus Temperature (615 c) AL 380
# T_S = scaler(T_S,500.0,919.0)                     #  K -Solidus Temperature (615 c) AL 380
# t_surr_s = temp_scaler(t_surr, temp_init, t_surr)
# t_surr = scaler(t_surr,500.0,919.0)
t_surr = props['t_surr']                   #  K -Surrounding Temperature (500 c) AL 380
T_lt = torch.tensor(T_L).float().to(device)    # Liquidus Temperature tensor
T_st = torch.tensor(T_S).float().to(device)    # Solidus Temperature tensor
t_surrt = torch.tensor(t_surr).float().to(device)   # Surrounding Temperature tensor

temp_var = {"T_st":T_st,"T_lt":T_lt,"t_surrt":t_surrt,"temp_init_t":temp_init_t}

# %% [markdown]
# ### Dataset Preparation for pytorch

# %%
train_inputs,test_inputs =train_test_split(input_t,test_size=0.2,random_state=42) # input data split
# print(train_inputs.shape)
tr_inp_pde,ts_inp_pde = train_test_split( inp_pdet,test_size=0.2,random_state=42) # input pde data split
# print(tr_inp_pde.shape)
tr_inp_ic,ts_inp_ic = train_test_split( inp_ict,test_size=0.2,random_state=42) # input ic data split
# print(tr_inp_ic.shape)

tr_inp_bcl,ts_inp_bcl = train_test_split( inp_bclt,test_size=0.2,random_state=42) # input bc left data split
tr_inp_bcr,ts_inp_bcr = train_test_split( inp_bclr,test_size=0.2,random_state=42) # input bc right data split
# nn
# 

train_temp,test_temp = train_test_split(temp_t,test_size=0.2,random_state=42) # output data split



# %%
class Data_Tensor_Dataset(TensorDataset):#dataset class for tsimulation data
    def __init__(self,inputs,outputs,transform=None, target_transform =None):   
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, index):
        return self.inputs[index],self.outputs[index]
    
    def __len__(self):
        return len(self.inputs)

class ResDataset(TensorDataset): #dataset class for pde residuals and bcs,ics
    def __init__(self, inputs,transform=None, target_transform =None):
        self.inputs = inputs
        

    def __getitem__(self, index):
        return self.inputs[index]
    
    def __len__(self):
        return len(self.inputs)

# %% [markdown]
# ### Dataset Preparation

# %%
inp_dataset = Data_Tensor_Dataset(train_inputs,train_temp)
inp_dataset_test = Data_Tensor_Dataset(test_inputs,test_temp)

inp_pde_dataset = ResDataset(tr_inp_pde) # pde residual dataset for training
inp_pde_dataset_test = ResDataset(ts_inp_pde) # pde residual dataset for testing

inp_ic_dataset = ResDataset(tr_inp_ic) # ic residual dataset for training
inp_ic_dataset_test = ResDataset(ts_inp_ic) # ic residual dataset for testing

inp_bcl_dataset = ResDataset(tr_inp_bcl) # bc left residual dataset for training
inp_bcl_dataset_test = ResDataset(ts_inp_bcl) # bc left residual dataset for testing

inp_bcr_dataset = ResDataset(tr_inp_bcr) # bc right residual dataset for training
inp_bcr_dataset_test = ResDataset(ts_inp_bcr)   # bc right residual dataset for testing

# %%
# print(len(inp_ic_dataset))

# %% [markdown]
# ### Dataloader Preparation

# %%
rand_smpl = RandomSampler(inp_dataset, replacement=False,num_samples=10000)  # random sampler for training/simulation data
rand_smpl_pde = RandomSampler(inp_pde_dataset, replacement=False, num_samples=10000) # random sampler for pde residuals-training
rand_smpl_ic = RandomSampler(inp_ic_dataset, replacement=False, num_samples=10000)  # random sampler for ic residuals-training
rand_smpl_bcl = RandomSampler(inp_bcl_dataset, replacement=False, num_samples=10000) # random sampler for bc left residuals-training
rand_smpl_bcr = RandomSampler(inp_bcr_dataset, replacement=False, num_samples=10000) # random sampler for bc right residuals-training

rand_smpl_test = RandomSampler(inp_dataset_test, replacement=False, num_samples=10000 )  # random sampler for testing/simulation data
rand_smpl_pde_test = RandomSampler(inp_pde_dataset_test,replacement=False, num_samples=10000)  # random sampler for pde residuals
rand_smpl_ic_test = RandomSampler(inp_ic_dataset_test,replacement=False, num_samples=10000 )  # random sampler for ic residuals
rand_smpl_bcl_test = RandomSampler(inp_bcl_dataset_test,replacement=False, num_samples=10000) # random sampler for bc left residuals
rand_smpl_bcr_test = RandomSampler(inp_bcr_dataset_test,replacement=False, num_samples=10000) # random sampler for bc right residuals

train_loader = DataLoader(inp_dataset, batch_size=256, sampler=rand_smpl) # training data loader
pde_loader = DataLoader(inp_pde_dataset, batch_size=256, sampler=rand_smpl_pde) # pde residual data loader training
ic_loader = DataLoader(inp_ic_dataset, batch_size=256, sampler=rand_smpl_ic) # ic residual data loader training
bcl_loader = DataLoader(inp_bcl_dataset, batch_size=256, sampler=rand_smpl_bcl) # bc left residual data loader training
bcr_loader = DataLoader(inp_bcr_dataset, batch_size=256, sampler=rand_smpl_bcr) # bc right residual data loader training


test_loader = DataLoader(inp_dataset_test, batch_size=256, sampler=rand_smpl_test) # testing data loader
pde_loader_test = DataLoader(inp_pde_dataset_test, batch_size=256, sampler=rand_smpl_pde_test)
ic_loader_test = DataLoader(inp_ic_dataset_test, batch_size=256, sampler=rand_smpl_ic_test)
bcl_loader_test = DataLoader(inp_bcl_dataset_test, batch_size=256, sampler=rand_smpl_bcl_test)
bcr_loader_test = DataLoader(inp_bcr_dataset_test, batch_size=256, sampler=rand_smpl_bcr_test)



input_size = 2
hidden_size = props['neurons'] # 64
output_size=1

learning_rate = props['learning_rate'] # 0.001
hidden_layers = props['layers']



epochs_1 = props['epochs'] # 1000

epochs_2 = 10
from Model.model import PINN

model = PINN(input_size, hidden_size, output_size,hidden_layers).to(device)
optimizer_1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer_2 = torch.optim.LBFGS(model.parameters(), lr=learning_rate)


torch.autograd.set_detect_anomaly(True)

#train_losses, test_losses, pde_losses, bc_losses,ic_losses, data_losses = training_loop(epochs_1, model, loss_fn_data, \
                #   optimizer_1,train_loader,pde_loader, ic_loader,\
                #   bcl_loader,bcr_loader,\
                #   test_loader,pde_loader_test,ic_loader_test,\
                #   bcl_loader_test,bcr_loader_test,\
                #   temp_var)  # Train the model 

loss_train,loss_test,best_model = training_loop(epochs_1, model, loss_fn_data, \
                  optimizer_1,train_loader,pde_loader, ic_loader,\
                  bcl_loader,bcr_loader,\
                  test_loader,pde_loader_test,ic_loader_test,\
                  bcl_loader_test,bcr_loader_test,\
                  temp_var)


torch.autograd.set_detect_anomaly(True)

#train_losses, test_losses, pde_losses, bc_losses,ic_losses, data_losses = training_loop(epochs_1, model, loss_fn_data, \
                #   optimizer_1,train_loader,pde_loader, ic_loader,\
                #   bcl_loader,bcr_loader,\
                #   test_loader,pde_loader_test,ic_loader_test,\
                #   bcl_loader_test,bcr_loader_test,\
                #   temp_var)  # Train the model 

loss_train,loss_test,best_model = training_loop(epochs_1, model, loss_fn_data, \
                  optimizer_1,train_loader,pde_loader, ic_loader,\
                  bcl_loader,bcr_loader,\
                  test_loader,pde_loader_test,ic_loader_test,\
                  bcl_loader_test,bcr_loader_test,\
                  temp_var)


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

# %%

# Parse job ID from arguments
parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=str, default="000000", help="Unique SLURM job ID")
args = parser.parse_args()

# Top-level output directory
output_root = "output_files"
if not os.path.exists(output_root):
    os.makedirs(output_root)
    print(f"Created root output directory: {output_root}")

# Subdirectory for this job
job_dir = os.path.join(output_root, f"job_{args.job_id}")
os.makedirs(job_dir, exist_ok=True)
print(f"Saving outputs to: {job_dir}")

# Save model
model_path = os.path.join(job_dir, "best-model.pth")
torch.save(best_model.state_dict(), model_path)

# Define paths for loss logs
loss_train_pth = os.path.join(job_dir, "train-loss.pkl")
loss_test_pth = os.path.join(job_dir, "test-loss.pkl")

# Save losses
with open(loss_train_pth, "wb") as f:
    pickle.dump(loss_train, f)

with open(loss_test_pth, "wb") as f:
    pickle.dump(loss_test, f)
# Job summary details (replace these with your actual variable values)
activation_fn = "Tanh"
hidden_layers = hidden_layers + 1
hidden_size = hidden_size

optimizer_used = optimizer_1
train_file_used = "simdata_mush_dirc.py"  # Or dynamically fetched
Model_file= os.path.basename(__file__)
job_id = args.job_id

# Summary content
summary_text = f"""
==================== JOB SUMMARY ====================
Job ID: {job_id}
Model Architecture: {hidden_size} x {hidden_layers}
Activation Function: {activation_fn}
Optimizer: {optimizer_used}
Learning Rate: {learning_rate}
Training Data Source: {train_file_used}
Model File: {Model_file}
Saved Files:
  - Model Weights: best-model.pth
  - Training Loss: train-loss.pkl
  - Testing Loss: test-loss.pkl
======================================================
"""

# Save to text file
summary_path = os.path.join(job_dir, "job_summary.txt")
with open(summary_path, "w") as f:
    f.write(summary_text)

print(f"Job summary saved at: {summary_path}")

# Final logs
print(f"Training loss saved at: {loss_train_pth}")
print(f"Testing loss saved at: {loss_test_pth}")
print("Training complete")
print(f"Model Architecture: {hidden_size} x {hidden_layers+1}, Learning Rate: {learning_rate}, Activation: Tanh, Optimizer: Adam")
