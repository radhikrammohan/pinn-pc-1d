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
k_mo = torch.tensor(props['k_mo'],dtype=torch.float32,device=device)


cp_l_t = torch.tensor(props['cp_l'],dtype=torch.float32,device=device)
           
cp_s_t = torch.tensor(props['cp_s'],dtype=torch.float32,device=device)
# cp_m =  (cp_l+cp_s)/2                 # Specific heat of mushy zone is taken as average of liquid and solid specific heat
# cp_m = cp
           # Thermal diffusivity
alpha_l_t = k_l_t / (rho_l_t * cp_l_t) 

alpha_s_t = k_s_t / (rho_s_t*cp_s_t)
alpha_s_t = alpha_s_t.clone().detach().to(dtype=torch.float32, device=device)

# # alpha_m = k_m / (rho_m * cp_m)          #`Thermal diffusivity in mushy zone is taken as average of liquid and solid thermal diffusivity`

L_fusion_t = torch.tensor(props['L_fusion'],dtype=torch.float32,device=device) # J/kg  # Latent heat of fusion of aluminum
#          # Thermal diffusivity

def temp_scaler(temp_data, temp_init, t_surr):
    temp_data = (temp_data - t_surr) / (temp_init - t_surr)
    return temp_data

temp_init = torch.tensor(props['temp_init'], dtype=torch.float32, device=device)  # Initial temperature (K)
temp_init_s = temp_scaler(temp_init, temp_init, props['t_surr'])  # Scaled Initial Temperature
t_surr = torch.tensor(props['t_surr'], dtype=torch.float32, device=device)  # Surrounding temperature (K)
t_surr_s = temp_scaler(t_surr, temp_init, t_surr)  # Scaled Surrounding Temperature
# temp_init = temp_init.clone().detach().to(dtype=torch.float32, device=device)

                   
T_St = torch.tensor(props['T_S'] ,dtype=torch.float32,device=device) # K- Solidus Temperature (550 C)

T_S_s = temp_scaler(T_St, temp_init, t_surr)  # Scaled Solidus Temperature
T_Lt = torch.tensor(props['T_L'] ,dtype=torch.float32,device=device) #  K -Liquidus Temperature (615 c) AL 380
T_L_s = temp_scaler(T_Lt, temp_init, t_surr)  # Scaled Liquidus Temperature

temp_l = props['die_temp_l']  # Left boundary temperature (K)
temp_r = props['die_temp_r']  # Right boundary temperature (K)

temp_l_s = temp_scaler(torch.tensor(temp_l, dtype=torch.float32, device=device), temp_init, t_surr)  # Scaled Left Boundary Temperature
temp_r_s = temp_scaler(torch.tensor(temp_r, dtype=torch.float32, device=device), temp_init, t_surr)  # Scaled Right Boundary Temperature
length = props['length']  # Length of the die (m)
length_s = torch.tensor(length, dtype=torch.float32, device=device)  # Scaled Length of the die


def kramp(temp,v1,v2,T_L,T_S):              # Function to calculate thermal conductivity in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    
    k_m = torch.where(temp > T_L, v1, torch.where(temp < T_S, v2, v2 + slope*(temp-T_S)))
    
        
    return k_m

def cp_ramp(temp,v1,v2,T_L,T_S):        # Function to calculate specific heat capacity in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    cp_m = torch.where(temp > T_L, v1, torch.where(temp < T_S, v2, v2 + slope*(temp-T_S)))
    cp_max  = torch.maximum(v1, v2)
    cp_s = cp_m / cp_max  # Normalizing specific heat capacity to maximum value
    return cp_s

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
    
    
    # Ste = (cp_ramp(u_pred,cp_l_t,cp_s_t,T_L,T_S)*(T_Lt- T_St) )/ L_fusion_t
    
    def Ste(u_pred):
        T_range = temp_init - t_surr
        L_fusion_s = L_fusion_t / T_range
        delta_T = T_L_s - T_S_s
        # Ste = (cp_ramp(u_pred,cp_l_t,cp_s_t,T_L_s,T_S_s)*delta_T)/ L_fusion_s
        Ste = (cp_ramp(u_pred,cp_l_t,cp_s_t,T_L_s,T_S_s)*(delta_T))
        return Ste
    
    
    
    def alpha_m(u_pred):
        alpha_m = kramp(u_pred,k_l_t,k_s_t,T_L_s,T_S_s) \
            / (rho_ramp(u_pred,rho_l_t,rho_s_t,T_L_s,T_S_s) \
                * cp_ramp(u_pred,cp_l_t,cp_s_t,T_L_s,T_S_s)) 
        return alpha_m
   
    
    residual = torch.zeros_like(u_pred).to(device)
    
    if mask_l.any():
       alpha_l_s = alpha_l_t * (t[mask_l].view(-1) / (x[mask_l].view(-1)**2))

       residual[mask_l] = u_t[mask_l].view(-1) -  u_xx[mask_l].view(-1) # Liquid phase
       
    if mask_s.any():
       alpha_s_s = alpha_s_t * (t[mask_s].view(-1) / (x[mask_s].view(-1)**2))
       
       residual[mask_s] = u_t[mask_s].view(-1) -  u_xx[mask_s].view(-1) # Solid phase
       
    if mask_m.any():
       c3 = (1+ 1/Ste(u_pred[mask_m]))
       alpha_m_s = alpha_m(u_pred[mask_m]) * (t[mask_m].view(-1) / (x[mask_m].view(-1)**2))
       
       residual[mask_m] = u_t[mask_m].view(-1) - (1/c3) * u_xx[mask_m].view(-1) # Mushy phase
       
    # residual = u_t - (u_xx) # Calculate the residual of the PDE

    resid_mean = torch.mean(torch.square(residual))
    # resid_mean = nn.MSELoss()(residual,torch.zeros_like(residual).to(device))
    # print(resid_mean.dtype)ß
    
    return resid_mean 

def boundary_loss(model,x,t,t_surr,t_init):
    
        
    u_pred = model(x,t)
    # bc = torch.where(t == 0, t_init, t_surr)
    # def bc_func(x,t,t_surr,t_init):
    #     bc = torch.where(t == 0, t_init, t_surr)
    #     ramp_mask = torch.logical_and(t > 0 , t < 0.000330226858600583)
    #     bc = torch.where(ramp_mask, (t_surr - t_init)/(0.000330226858600583)*t, bc)

    #     bc = torch.where(t > 0.000330226858600583, t_surr, bc)
    #     return bc

    # bc_cal = bc_func(x,t,t_surr,t_init)
    
    # bc_mean =  torch.mean(torch.square(u_pred-bc_cal))
    # print(f"Boundary condition loss calculated: {u_pred.mean():.6f}")
    t_surr_c = torch.full_like(u_pred, t_surr)
    bc_mean =  torch.mean(torch.square(u_pred-t_surr_c))
    # bc_mean =  torch.mean(torch.square(u_pred-bc))
    # bc_mean = nn.MSELoss()(u_pred,bc)
   
    return bc_mean

def ic_loss(model,x,t,temp_init):
    
    u_pred = model(x,t)
    
    # def ic_func(x,t,temp_init):
    #     return temp_init
    
    # u_ic = ic_func(x,t,temp_init)
    
    # # u_del = u_pred - temp_init
    # temp_i = torch.full_like(u_pred,temp_init)

    lin_temp = temp_l_s + (temp_l_s - temp_r_s) * x
    mag = temp_init - temp_l_s
    dome  = mag  * torch.sin( torch.pi * x)
    temp_i  = lin_temp + dome

    # ic_mean = nn.MSELoss()(u_pred,temp_i)    
    ic_mean = torch.mean(torch.square(u_pred-temp_i))
    # print(f"Initial condition loss calculated: {u_pred.mean():.6f}")
    return ic_mean

def accuracy(u_pred, u_true):
    return torch.mean(torch.abs(u_pred - u_true))


def pde_resid(model,x,t,T_S,T_L):
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


    # Ste = (cp_ramp(u_pred,cp_l_t,cp_s_t,T_L_s,T_S_s)*(T_Lt- T_St))/ L_fusion_t

    def Ste(u_pred):
        Ste = (cp_ramp(u_pred,cp_l_t,cp_s_t,T_L_s,T_S_s)*(T_Lt- T_St))/ L_fusion_t
        return Ste
    
    
    
    def alpha_m(u_pred):
        alpha_m = kramp(u_pred,k_l_t,k_s_t,T_L_s,T_S_s) \
            / (rho_ramp(u_pred,rho_l_t,rho_s_t,T_L_s,T_S_s) \
                * cp_ramp(u_pred,cp_l_t,cp_s_t,T_L_s,T_S_s)) 
        return alpha_m
   
    
    residual = torch.zeros_like(u_pred).to(device)
    
   
    if mask_l.any():
        residual[mask_l] = u_t[mask_l].view(-1) - alpha_l_t * u_xx[mask_l].view(-1) # Liquid phase
        # print("Liquid phase residual calculated")
    if mask_s.any():
        residual[mask_s] = u_t[mask_s].view(-1) - alpha_s_t * u_xx[mask_s].view(-1) # Solid phase
        # print("Solid phase residual calculated")
    if mask_m.any():
        c3 = (1+ 1/Ste(u_pred[mask_m]))
        residual[mask_m] = u_t[mask_m].view(-1) - (alpha_m(u_pred[mask_m]) /c3) * u_xx[mask_m].view(-1) # Mushy phase
        # print("Mushy phase residual calculated")

    # residual = u_t - (u_xx) # Calculate the residual of the PDE
   
    
    
    return residual
