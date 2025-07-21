# %% [markdown]
# # Import models

# %%
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
import optuna


current_dir = os.path.dirname(os.path.abspath(__file__))
    
training_data_dir = os.path.join(current_dir, '../training_data')
model_dir = os.path.join(current_dir, '../')
sys.path.insert(0,str(training_data_dir))
sys.path.insert(0,str(model_dir))

from simdata_mush_dirc_icc import  *
from Model.loss_func_sol_norm import loss_fn_data,pde_loss,ic_loss,boundary_loss
from Model.train_testloop_norm import *

def set_seed(seed):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Use deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # This may not exist on macOS if not using GPU or CuDNN
    if torch.backends.mps.is_available():  # MPS = Apple GPU backend
        print("Using Apple Metal Performance Shaders (MPS)")
    elif torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(1234)


# %% [markdown]
# # Get the training data

# %%

settings_path_1 = os.path.join(current_dir, '..', 'training_data', 'settings.json')
with open(settings_path_1,'r') as file:
    settings = json.load(file)


heat_data = HT_sim(settings)
alpha = heat_data.alpha_l
tempfield = heat_data.datagen()

heat_data.plot_temp(32)
dt = heat_data.dt
dx = heat_data.dx
# print(heat_data.dx)
# print(dt)
settings_path = os.path.join(current_dir, '..', 'training_data', 'settings.json')
with open(settings_path,'r') as file:
    props = json.load(file)
    


# %%
# plot the intial temperature field
plt.figure(figsize=(8, 6))
plt.plot( tempfield[0, :], label='Initial Temperature Field')
plt.title('Initial Temperature Field')
plt.xlabel('Position (x)')
plt.ylabel('Temperature')
plt.legend()
plt.grid()


# %%
temp_data = tempfield.flatten()

def temp_scaler(temp_data, temp_init, t_surr):
    temp_data = (temp_data - t_surr) / (temp_init - t_surr)
    return temp_data
temp_init = props['temp_init']
t_surr = props['t_surr']
# temp_data = scaler(temp_data,400.0,919.0)

temp_data_s = temp_scaler(temp_data, temp_init, t_surr)

print(temp_data_s.max())

# %%
num_steps = tempfield.shape[0]
numpoints = tempfield.shape[1] 

pde_pts= props["pde_pts"]
ic_pts = props["ic_pts"]
bc_pts = props["bc_pts"]

length = props['length']
time_end = props['time_end']

x_c = 1/length
k_l = props['k_l']
k_s = props['k_s']
rho_l = props['rho_l']
rho_s = props['rho_s']
cp_l = props['cp_l']
cp_s = props['cp_s']
alpha_l = k_l / (rho_l * cp_l)
alpha_s = k_s / (rho_s * cp_s)

alpha_max = max(alpha_l, alpha_s)
t_c = (alpha_max/(length**2))
temp_c = props['temp_init']

inp_data = fdd(15e-3, time_end, numpoints, num_steps)


def scale2(x,x_c,t_c):
    scaled_x = x.copy()
    scaled_x[:,0] = x[:,0] * x_c
    scaled_x[:,1] = x[:,1] * t_c
    return scaled_x

inp_data2 = scale2(inp_data,x_c,t_c)

# input dataset-pde residual
# The pde inputs are generated using the pdeinp function in simdata.py
pde_data = pdeinp(dx,length-dx,dt,time_end,pde_pts,"Sobol",scl="False") 

pde_data2 = scale2(pde_data,x_c,t_c)

# input dataset - ic residual
ic_data = icinp(length,ic_pts,scl="False")
ic_data2 = scale2(ic_data,x_c,t_c)
# input dataset - boundary residual
bc_ldata = bcinp(length,time_end,bc_pts,dt,scl="False")[0]
bc_rdata = bcinp(length,time_end,bc_pts,dt,scl="False")[1]

bc_ldata2 = scale2(bc_ldata,x_c,t_c)
bc_rdata2 = scale2(bc_rdata,x_c,t_c)


# %%
t_lim = 1.5
pde_data_new = pde_act_pts(dx,length-dx,dt,time_end,t_lim,pde_pts,"Sobol",scl="False") 
pde_data_new2 = scale2(pde_data_new,x_c,t_c)

# %% [markdown]
# # Prepare  the Inputs

# %%

if torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

# print('Using device:', device)

# %% [markdown]
# ### Tensor inputs

input_t = torch.tensor(inp_data2).float().to(device)
inp_pdet = torch.tensor(pde_data2).float().to(device)
inp_ict = torch.tensor(ic_data2).float().to(device)
inp_bclt = torch.tensor(bc_ldata2).float().to(device)
inp_bclr = torch.tensor(bc_rdata2).float().to(device)



temp_t = torch.tensor(temp_data_s).float().to(device)
temp_t = temp_t.view(-1,1)

temp_init_s= temp_scaler(919.0, temp_init, t_surr)
# temp_init = scaler(temp_init,500.0,919.0)

temp_init_t = torch.tensor(temp_init_s).float().to(device)
T_L = props['T_L']                   #  K -Liquidus Temperature (615 c) AL 380
T_L_s = temp_scaler(T_L,temp_init, t_surr)                     #  K -Liquidus Temperature (615 c) AL 380
# T_L = scaler(T_L,500.0,919.0)
T_S = props['T_S']                   #  K -Solidus Temperature (615 c) AL 380
T_S_s = temp_scaler(T_S,temp_init, t_surr)                     #  K -Solidus Temperature (615 c) AL 380
# T_S = scaler(T_S,500.0,919.0)                     #  K -Solidus Temperature (615 c) AL 380

# t_surr = scaler(t_surr,500.0,919.0)
t_surr = props['t_surr']                   #  K -Surrounding Temperature (500 c) AL 380
t_surr_s = temp_scaler(t_surr, temp_init, t_surr)
T_lt = torch.tensor(T_L_s).float().to(device)    # Liquidus Temperature tensor
T_st = torch.tensor(T_S_s).float().to(device)    # Solidus Temperature tensor
t_surrt = torch.tensor(t_surr_s).float().to(device)   # Surrounding Temperature tensor

die_left = props['die_temp_l']
die_left = temp_scaler(die_left, temp_init, t_surr) # die left temperature
die_right = props['die_temp_r']
die_right = temp_scaler(die_right, temp_init, t_surr) # die right temperature
temp_var = {"T_st":T_st,"T_lt":T_lt,"t_surrt":t_surrt,"temp_init_t":temp_init_t,\
               "die_temp_l":die_left,"die_temp_r":die_right} # temperature variables dictionary
print(temp_var)
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


class PINNLoaderFactory:
    def __init__(self, data_dict, sample_size):
        """
        Expects a dict with the following keys:
        - 'train_inputs', 'train_temp', 'test_inputs', 'test_temp'
        - 'tr_inp_pde', 'ts_inp_pde', ...
        """
        self.data = data_dict
        self.sample_size = sample_size
        self._build_datasets()

    def _build_datasets(self):
        self.inp_dataset       = Data_Tensor_Dataset(self.data['train_inputs'], self.data['train_temp'])
        self.inp_dataset_test  = Data_Tensor_Dataset(self.data['test_inputs'], self.data['test_temp'])

        self.inp_pde_dataset       = ResDataset(self.data['tr_inp_pde'])
        self.inp_pde_dataset_test  = ResDataset(self.data['ts_inp_pde'])
        self.inp_ic_dataset        = ResDataset(self.data['tr_inp_ic'])
        self.inp_ic_dataset_test   = ResDataset(self.data['ts_inp_ic'])
        self.inp_bcl_dataset       = ResDataset(self.data['tr_inp_bcl'])
        self.inp_bcl_dataset_test  = ResDataset(self.data['ts_inp_bcl'])
        self.inp_bcr_dataset       = ResDataset(self.data['tr_inp_bcr'])
        self.inp_bcr_dataset_test  = ResDataset(self.data['ts_inp_bcr'])

    def _sampler(self, dataset):
        return RandomSampler(dataset, replacement=False, num_samples=min(self.sample_size, len(dataset)))

    def get_loaders(self, batch_size):
        return {
            'train_loader': DataLoader(self.inp_dataset, batch_size=batch_size, sampler=self._sampler(self.inp_dataset)),
            'pde_loader': DataLoader(self.inp_pde_dataset, batch_size=batch_size, sampler=self._sampler(self.inp_pde_dataset)),
            'ic_loader': DataLoader(self.inp_ic_dataset, batch_size=batch_size, sampler=self._sampler(self.inp_ic_dataset)),
            'bcl_loader': DataLoader(self.inp_bcl_dataset, batch_size=batch_size, sampler=self._sampler(self.inp_bcl_dataset)),
            'bcr_loader': DataLoader(self.inp_bcr_dataset, batch_size=batch_size, sampler=self._sampler(self.inp_bcr_dataset)),

            'test_loader': DataLoader(self.inp_dataset_test, batch_size=batch_size, sampler=self._sampler(self.inp_dataset_test)),
            'pde_loader_test': DataLoader(self.inp_pde_dataset_test, batch_size=batch_size, sampler=self._sampler(self.inp_pde_dataset_test)),
            'ic_loader_test': DataLoader(self.inp_ic_dataset_test, batch_size=batch_size, sampler=self._sampler(self.inp_ic_dataset_test)),
            'bcl_loader_test': DataLoader(self.inp_bcl_dataset_test, batch_size=batch_size, sampler=self._sampler(self.inp_bcl_dataset_test)),
            'bcr_loader_test': DataLoader(self.inp_bcr_dataset_test, batch_size=batch_size, sampler=self._sampler(self.inp_bcr_dataset_test)),
        }



# %% [markdown]
# # Prepare the Model

# %%
input_size = 2
hidden_size = 45#  best yet is 8
output_size=1

learning_rate = 0.002 # 0.001
hidden_layers = 3  #best yet is 4


# epochs_1 = props['epochs'] # 1000
epochs_1 =1000
epochs_2 = 100
from Model.model import PINN

model = PINN(input_size, hidden_size, output_size,hidden_layers).to(device)
optimizer_1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer_2 = torch.optim.LBFGS(model.parameters(), lr=learning_rate,line_search_fn ='strong_wolfe')



# %%


# %% [markdown]
# # Train the model

# %%
torch.autograd.set_detect_anomaly(True)


# loss_train,loss_test,best_model = training_loop(epochs_1, model, loss_fn_data, \
#                   optimizer_1,train_loader,pde_loader, ic_loader,\
#                   bcl_loader,bcr_loader,\
#                   test_loader,pde_loader_test,ic_loader_test,\
#                   bcl_loader_test,bcr_loader_test,\
#                   temp_var)

# loss_train,loss_test,best_model = training_loop(epochs_1, model, loss_fn_data, \
#                   optimizer_2,train_loader,pde_loader, ic_loader,\
#                   bcl_loader,bcr_loader,\
#                   test_loader,pde_loader_test,ic_loader_test,\
#                   bcl_loader_test,bcr_loader_test,\
#                   temp_var)

# %% [markdown]
# # Collect the results and store it an folder/ Visualise the results

# %%
def objective(trial):
    hidden_size = trial.suggest_int('hidden_size', 1, 10)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
    batch_size = trial.suggest_int('batch_size', 32, 512)
    epochs_1 = 1000

    model = PINN(input_size, hidden_size, output_size, hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Dataloader with the suggested batch size
    
    data_dict = {
    'train_inputs': train_inputs,
    'train_temp': train_temp,
    'test_inputs': test_inputs,
    'test_temp': test_temp,
    'tr_inp_pde': tr_inp_pde,
    'ts_inp_pde': ts_inp_pde,
    'tr_inp_ic': tr_inp_ic,
    'ts_inp_ic': ts_inp_ic,
    'tr_inp_bcl': tr_inp_bcl,
    'ts_inp_bcl': ts_inp_bcl,
    'tr_inp_bcr': tr_inp_bcr,
    'ts_inp_bcr': ts_inp_bcr,
}
    
    data_loader = PINNLoaderFactory(data_dict=data_dict,
        sample_size=10000
    )
    
    train_dataloader = data_loader.get_loaders(batch_size)['train_loader']
    train_loader_pde = data_loader.get_loaders(batch_size)['pde_loader']
    train_loader_init = data_loader.get_loaders(batch_size)['ic_loader']
    train_loader_bc_l = data_loader.get_loaders(batch_size)['bcl_loader']
    train_loader_bc_r = data_loader.get_loaders(batch_size)['bcr_loader']
    
    test_dataloader = data_loader.get_loaders(batch_size)['test_loader']
    pde_test_dataloader = data_loader.get_loaders(batch_size)['pde_loader_test']
    ic_test_dataloader = data_loader.get_loaders(batch_size)['ic_loader_test']
    test_bc_l_dataloader = data_loader.get_loaders(batch_size)['bcl_loader_test']
    test_bc_r_dataloader = data_loader.get_loaders(batch_size)['bcr_loader_test']

    
    # Trainig Loop
    train_loss = 0.0
    data_loss_b = 0.0
    phy_loss_acc = 0.0
    init_loss_acc = 0.0
    bc_loss_acc = 0.0
    
    test_loss = 0.0
    data_loss_t = 0.0
    phy_loss_t = 0.0
    ic_loss_t = 0.0
    bc_l_loss_t = 0.0

    
    for epoch in range(epochs_1):
        model.train()
        for (batch, batch_pde, batch_init, batch_left, batch_right) in \
             zip_longest(train_dataloader, train_loader_pde, train_loader_init, train_loader_bc_l, train_loader_bc_r):
            
            if batch is None or batch_pde is None or batch_init is None or batch_left is None or batch_right is None:
                continue
            
            inputs, temp_inp = batch
            inputs_pde = batch_pde
            inputs_init = batch_init
            inputs_left = batch_left
            inputs_right = batch_right

            inputs, temp_inp = inputs, temp_inp
            inputs_pde = inputs_pde
            inputs_init = inputs_init
            inputs_left = inputs_left
            inputs_right = inputs_right
            
            
            optimizer.zero_grad()  # Zero the gradients before backpropagation
            
            # Forward pass for data prediction
            u_pred_d = model(inputs[:, 0].unsqueeze(1), inputs[:, 1].unsqueeze(1))
            data_loss = loss_fn_data(u_pred_d, temp_inp)  # Data loss
            
            # Forward pass for initial condition prediction
            # u_initl = model(inputs_init[:, 0].unsqueeze(1), inputs_init[:, 1].unsqueeze(1))
            init_loss = ic_loss(model,inputs_init[:, 0].unsqueeze(1), inputs_init[:, 1].unsqueeze(1),temp_init_t)  # Initial condition loss
            
            # Forward pass for boundary conditions
            # u_left = model(inputs_left[:, 0].unsqueeze(1), inputs_left[:, 1].unsqueeze(1))
            # u_right = model(inputs_right[:, 0].unsqueeze(1), inputs_right[:, 1].unsqueeze(1))
            
            # Boundary condition loss (left and right)
            bc_loss_left = boundary_loss(model, inputs_left[:, 0].unsqueeze(1), inputs_left[:, 1].unsqueeze(1), die_left,temp_init_t)
            bc_loss_right = boundary_loss(model, inputs_right[:, 0].unsqueeze(1), inputs_right[:, 1].unsqueeze(1), die_right,temp_init_t)
            bc_loss = 0.5*(bc_loss_left + bc_loss_right)
            # Calculate individual losses
            phy_loss = pde_loss(model, inputs_pde[:, 0].unsqueeze(1), inputs_pde[:, 1].unsqueeze(1), T_st, T_lt)  # PDE loss

            # Define weights for the different losses
            w0, w1, w2, w3 = 1, 1, 1, 1
            # Calculate total loss
            # loss = data_loss 
            loss =  w1 * phy_loss + w2 * init_loss + w3 * bc_loss
            # Backpropagation
            loss.backward(retain_graph=True)  # Backpropagate the gradients

            def closure():
                optimizer.zero_grad()
                # Forward pass for data prediction
                u_pred_d = model(inputs[:, 0].unsqueeze(1), inputs[:, 1].unsqueeze(1))
                data_loss = loss_fn_data(u_pred_d, temp_inp)  # Data loss
                
                # Forward pass for initial condition prediction
                # u_initl = model(inputs_init[:, 0].unsqueeze(1), inputs_init[:, 1].unsqueeze(1))
                init_loss = ic_loss(model,inputs_init[:, 0].unsqueeze(1), inputs_init[:, 1].unsqueeze(1),temp_init_t)  # Initial condition loss
                
                # Forward pass for boundary conditions
                # u_left = model(inputs_left[:, 0].unsqueeze(1), inputs_left[:, 1].unsqueeze(1))
                # u_right = model(inputs_right[:, 0].unsqueeze(1), inputs_right[:, 1].unsqueeze(1))
                
                # Boundary condition loss (left and right)
                bc_loss_left = boundary_loss(model, inputs_left[:, 0].unsqueeze(1), inputs_left[:, 1].unsqueeze(1), die_left,temp_init_t)
                bc_loss_right = boundary_loss(model, inputs_right[:, 0].unsqueeze(1), inputs_right[:, 1].unsqueeze(1), die_right,temp_init_t)
                bc_loss = 0.5*(bc_loss_left + bc_loss_right)
                # Calculate individual losses
                phy_loss = pde_loss(model, inputs_pde[:, 0].unsqueeze(1), inputs_pde[:, 1].unsqueeze(1), T_st, T_lt)  # PDE loss

                # Define weights for the different losses
                w0, w1, w2, w3 = 1, 1, 1, 1
                # Calculate total loss
                loss = w1 * phy_loss + w2 * init_loss + w3 * bc_loss
                loss.backward(retain_graph=True)
                return loss
            
            if optimizer.__class__ == torch.optim.Adam:
                optimizer.step()  # Update the weights
            else:
                optimizer.step(closure)
             
            # Accumulate losses for tracking
            train_loss += loss.item()
            
            data_loss_b += data_loss.item()
            phy_loss_acc += phy_loss.item()
            init_loss_acc += init_loss.item()
            bc_loss_acc += bc_loss.item()
            
            # Average the losses

        train_loss /= len(train_dataloader)
        data_loss_b /= len(train_dataloader)
        phy_loss_acc /= len(train_loader_pde)
        init_loss_acc /= len(train_loader_init)
        bc_loss_acc /= len(train_loader_bc_l)
        model.eval()
        test_loss = 0
        
        # Evaluate on test data without gradient calculation
        for (batch, batch_pde, batch_init, batch_left, batch_right) in zip_longest(test_dataloader, \
            pde_test_dataloader, ic_test_dataloader, test_bc_l_dataloader, test_bc_r_dataloader):
            
            if batch is None or batch_pde is None or batch_init is None or batch_left is None or batch_right is None:
               continue  # Skip this iteration
            inputs, temp_inp = batch
            inputs_pde = batch_pde
            inputs_init = batch_init
            inputs_left = batch_left
            inputs_right = batch_right
            
            inputs, temp_inp = inputs, temp_inp
            inputs_pde = inputs_pde
            inputs_init = inputs_init
            inputs_left = inputs_left
            inputs_right = inputs_right
            
            
            u_pred = model(inputs[:, 0].unsqueeze(1), inputs[:, 1].unsqueeze(1))
            data_loss_t = loss_fn_data(u_pred, temp_inp)
            
            # u_initl = model(inputs_init[:, 0].unsqueeze(1), inputs_init[:, 1].unsqueeze(1))
            init_loss_t = ic_loss(model,inputs_init[:, 0].unsqueeze(1), inputs_init[:, 1].unsqueeze(1),temp_init_t)
            
            # u_left = model(inputs_left[:, 0].unsqueeze(1), inputs_left[:, 1].unsqueeze(1))
            # u_right = model(inputs_right[:, 0].unsqueeze(1), inputs_right[:, 1].unsqueeze(1))
            
            bc_loss_left_t = boundary_loss(model, inputs_left[:, 0].unsqueeze(1), inputs_left[:, 1].unsqueeze(1),die_left,temp_init_t)
            bc_loss_right_t = boundary_loss(model, inputs_right[:, 0].unsqueeze(1), inputs_right[:, 1].unsqueeze(1),die_right,temp_init_t)
            bc_loss_t = 0.5*(bc_loss_left_t + bc_loss_right_t)
            
            phy_loss_t = pde_loss(model, inputs_pde[:, 0].unsqueeze(1), inputs_pde[:, 1].unsqueeze(1), T_st, T_lt)
            
            w0, w1, w2, w3 = 1,1,1,1
            # loss_t =  data_loss_t 
            loss_t = w1 * phy_loss_t + w2 * init_loss_t + w3 * bc_loss_t

            test_loss += loss_t.item()
            data_loss_t += data_loss_t.item()
            phy_loss_t += phy_loss_t.item()
            ic_loss_t += init_loss_t.item()
            bc_l_loss_t += bc_loss_t.item()
            
        test_loss /= len(test_dataloader)
        data_loss_t /= len(test_dataloader)
        phy_loss_t /= len(test_dataloader)
        ic_loss_t /= len(test_dataloader)
        bc_l_loss_t /= len(test_dataloader)
            
        
        
        
    

    return test_loss  # Return the last value of the test loss as the objective value

# %%
# Optional: set a name and direction
study = optuna.create_study(direction='minimize', study_name='PINN Hyperparameter Tuning')

# Start optimization
study.optimize(objective, n_trials=100, timeout=3600)  # 100 trials or 1 hour

# %%
print("Best trial:")
print("  Value: ", study.best_trial.value)
print("  Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# %%
# optuna.visualization.plot_optimization_history(study).show()
# optuna.visualization.plot_param_importances(study).show()


