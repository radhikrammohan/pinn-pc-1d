import json
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn import svm
import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

if torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

print('Using device:', device)

json_path = os.path.join(os.path.dirname(__file__), '../training_data/settings.json')
with open(json_path) as json_file:
    props = json.load(json_file)

# Material properties

rho_l_t = torch.tensor(props['rho_l'],dtype=torch.float32,device=device)
                   # Density of AL380 (kg/m^3)
rho_s_t = torch.tensor(props['rho_s'],dtype=torch.float32,device=device)

# rho_m = (rho_l + rho_s )/2       # Desnity in mushy zone is taken as average of liquid and solid density

                      # W/m-K
k_l_t = torch.tensor(props['k_l'],dtype=torch.float32,device=device)
                  # W/m-K
k_s_t = torch.tensor(props['k_s'],dtype=torch.float32,device=device)
# k_m =  (k_l+k_s)/2                     # W/m-K
k_mo = torch.tensor(props['k_m'],dtype=torch.float32,device=device)


cp_l_t = torch.tensor(props['cp_l'],dtype=torch.float32,device=device)
           
cp_s_t = torch.tensor(props['cp_s'],dtype=torch.float32,device=device)
# cp_m =  (cp_l+cp_s)/2                 # Specific heat of mushy zone is taken as average of liquid and solid specific heat
# cp_m = cp
           # Thermal diffusivity
alpha_l_t = k_l_t / (rho_l_t * cp_l_t) 

alpha_s_t = k_s_t / (rho_s_t*cp_s_t)
alpha_s_t = torch.tensor(alpha_s_t,dtype=torch.float32,device=device)

# # alpha_m = k_m / (rho_m * cp_m)          #`Thermal diffusivity in mushy zone is taken as average of liquid and solid thermal diffusivity`

L_fusion_t = torch.tensor(props['L_fusion'],dtype=torch.float32,device=device) # J/kg  # Latent heat of fusion of aluminum
#          # Thermal diffusivity

# t_surr = 500.0 
# temp_init = 919.0
                   
T_St = torch.tensor(props['T_S'] ,dtype=torch.float32,device=device) # K- Solidus Temperature (550 C)
T_Lt = torch.tensor(props['T_L'] ,dtype=torch.float32,device=device) #  K -Liquidus Temperature (615 c) AL 380



def kramp(temp,v1,v2,T_L,T_S):              # Function to calculate thermal conductivity in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    
    k_m = torch.where(temp > T_L, v1, torch.where(temp < T_S, v2, v2 + slope*(temp-T_S)))
    
        
    return k_m

def cp_ramp(temp,v1,v2,T_L,T_S):        # Function to calculate specific heat capacity in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    cp_m = torch.where(temp > T_L, v1, torch.where(temp < T_S, v2, v2 + slope*(temp-T_S)))
    
    return cp_m

def rho_ramp(temp,v1,v2,T_L,T_S):         # Function to calculate density in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    rho_m = torch.where(temp > T_L, v1, torch.where(temp < T_S, v2, v2 + slope*(temp-T_S)))
    
    return rho_m


def loss_fn_data(u_pred, u_true):
    return nn.MSELoss()(u_pred, u_true)

def l1_regularization(model, lambd):
    l1_reg = sum(param.abs().sum() for param in model.parameters())
    return l1_reg * lambd

def pde_loss(model,x,t,T_S,T_L):
    # u_pred.requires_grad = True
    x.requires_grad = True
    t.requires_grad = True
    
    u_pred = model(x,t).to(device)
    # u_pred  = model

    u_t = torch.autograd.grad(u_pred, t, 
                                torch.ones_like(u_pred),
                                create_graph=True,
                                allow_unused=True,
                                )[0] # Calculate the first time derivative
    if u_t is None:
        raise RuntimeError("u_t is None") # Check if u_t is None

    u_x = torch.autograd.grad(u_pred, 
                                x, 
                                torch.ones_like(u_pred), 
                                create_graph=True,
                                allow_unused =True)[0] # Calculate the first space derivative

    if u_x is None:
        raise RuntimeError("u_x is None") # Check if u_x is None
           
    u_xx = torch.autograd.grad(u_x, 
                                x, 
                                torch.ones_like(u_x), 
                                create_graph=True,
                                allow_unused=True,
                                materialize_grads=True)[0]
    
    if u_xx is None:
        raise RuntimeError("u_xx is None") # Check if u_xx is None

    # T_S_tensor = T_S.clone().detach().to(device)
    # T_L_tensor = T_L.clone().detach().to(device)
    
    mask_l = u_pred > T_L
    mask_s = u_pred < T_S
    mask_m = (u_pred <= T_L) & (u_pred >= T_S)
    
    c1 = rho_l_t * cp_l_t
    c2 = rho_s_t * cp_s_t
    Ste = (cp_ramp*(T_Lt- T_St) )/ L_fusion_t
    c3 = rho_ramp * cp_ramp(1+ 1/Ste)
    
    
    alpha_m = kramp(u_pred,k_l_t,k_s_t,T_L,T_S) \
        / (rho_ramp(u_pred,rho_l_t,rho_s_t,T_L,T_S) \
            * cp_ramp(u_pred,cp_l_t,cp_s_t,T_L,T_S)) 
    
    residual = torch.zeros_like(u_pred).to(device)
    
    residual[mask_l] = c1*u_t - alpha_l_t * u_xx[mask_l] # Liquid phase
    residual[mask_s] = c2*u_t - alpha_s_t * u_xx[mask_s] # Solid phase
    residual[mask_m] = c3*u_t - alpha_m * u_xx[mask_m] # Mushy phase
        
    
    # residual = u_t - (u_xx) # Calculate the residual of the PDE
   
    resid_mean = torch.mean(torch.square(residual))
    # resid_mean = nn.MSELoss()(residual,torch.zeros_like(residual).to(device))
    # print(resid_mean.dtype)
    
    return resid_mean

def boundary_loss(model,x,t,t_surr,t_init):
    
    # x.requires_grad = True
    # t.requires_grad = True
    # t_surr_t = torch.tensor(t_surr, device=device)
    # u_pred = model(x,t).requires_grad_(True)
    # u_x = torch.autograd.grad(u_pred,x, 
    #                             torch.ones_like(u_pred).to(device), 
    #                             create_graph=True,
    #                             allow_unused =True)[0] # Calculate the first space derivative
   
    # htc =10.0
    # if u_x is None:
    #     raise RuntimeError("u_x is None")
    # if u_pred is None:
    #     raise RuntimeError("u_pred is None")
    # if t_surr_t is None:
    #     raise RuntimeError("t_surr_t is None")
    # res_l = u_x -(htc*(u_pred-t_surr_t))
    
    # t_surr_t = t_surr.clone().detach().to(device)
    
    # def bc_func(x,t,t_surr,t_init):
    #     if (t == 0).any():
    #         return t_init
    #     else:
    #         return t_surr
        
    u_pred = model(x,t)
    bc = torch.where(t == 0, t_init, t_surr)
    
    bc_mean =  torch.mean(torch.square(u_pred-bc))
    # bc_mean = nn.MSELoss()(u_pred,bc)
   
    return bc_mean

def ic_loss(model,x,t,temp_init):
    
    u_pred = model(x,t)
    
    # def ic_func(x,t,temp_init):
    #     return temp_init
    
    # u_ic = ic_func(x,t,temp_init)
    
    # # u_del = u_pred - temp_init
    temp_i = torch.full_like(u_pred,temp_init)
   
    # ic_mean = nn.MSELoss()(u_pred,temp_i)    
    ic_mean = torch.mean(torch.square(u_pred-temp_i))
    return ic_mean

def accuracy(u_pred, u_true):
    return torch.mean(torch.abs(u_pred - u_true))