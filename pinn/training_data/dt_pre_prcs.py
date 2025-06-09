# import libraries

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

# file imports

try:
    current_dir = os.getcwd()
except:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
training_data_dir = os.path.join(current_dir, '../training_data')
model_dir = os.path.join(current_dir, '../')
sys.path.insert(0,str(training_data_dir))
sys.path.insert(0,str(model_dir))

from simdata_mush_dirc import *



class DataPreprocessor:
    def __init__(self, settings_path):
        
        self.settings_path = os.path.join(current_dir, '..', 'training_data', 'settings.json')
        with open(self.settings_path,'r') as file:
            self.settings = json.load(file)

        heat_data = HT_sim(self.settings)
        
        
        self.dt = heat_data.dt
        self.dx = heat_data.dx

        props_path = os.path.join(current_dir, '..', 'training_data', 'settings.json')
        with open(props_path,'r') as file:
            self.props = json.load(file)

        self.temp_init = self.props['temp_init']
        self.t_surr = self.props['t_surr']
        
        self.pde_pts = self.props['pde_pts']
        self.ic_pts = self.props['ic_pts']
        self.bc_pts = self.props['bc_pts']
        
        self.length = self.props['length']
        self.time_end = self.props['time_end']

        self.x_c = 1/ self.length

        self.k_max = np.max(self.props['k_l'],self.props['k_s'])
        self.rho_max = np.max(self.props['rho_l'],self.props['rho_s'])
        self.cp_max = np.max(self.props['cp_l'],self.props['cp_s'])
        self.alpha_max = self.k_max / (self.rho_max * self.cp_max)

        self.t_c = (self.alpha_max /(self.length**2))
        self.temp_c = self.props['temp_init']
        
    def scale2(x,x_c,t_c):
        scaled_x = x.copy()
        scaled_x[:,0] = x[:,0] * x_c
        scaled_x[:,1] = x[:,1] * t_c
        return scaled_x    

# get tempfield from sim_data file

    def get_tempfield(self):
        
        heat_data = HT_sim(self.settings)
        tempfield = heat_data.tempfield
        tempfield = tempfield.flatten()
        return tempfield
    
# scale the tempfield
       
    def temp_scaler(self,temp_data, temp_init, t_surr):
        temp_data = (temp_data - t_surr) / (temp_init - t_surr)
        return temp_data

    def tempdata_scaled(self):
        temp_init = self.props['temp_init']
        t_surr = self.props['t_surr']

        temp_data_scaled = self.temp_scaler(self.get_tempfield(), temp_init, t_surr)
        return temp_data_scaled
    
    


##  inputs for data points

    

## inputs for pde points
    def get_pde_inputs(self):
        pde_data = pde_inp(self.dx, (self.length-self.dx), self.dt, self.time_end, self.pde_pts, "Sobol",scl="False")
        return pde_data
    
    def pde_scaler(self):
        pde_data_scaled = self.scale2(self.get_pde_inputs(), self.x_c, self.t_c)

        return pde_data_scaled
    
    

##  inputs for ic points
    def get_ic_inputs(self):
        ic_data = icinp(self.length, self.ic_pts, scl="False")
        return ic_data

    def ic_scaled(self):
        ic_data_scaled = self.scale2(self.get_ic_inputs(), self.x_c, self.t_c)
        return ic_data_scaled
    
##  inputs for bc points
    def get_bc_l(self):
        bc_data_l = bcinp(self.length,self.time_end,self.bc_pts,self.dt,scl="False")[0]
        return bc_data_l

    def get_bc_r(self):
        bc_data_r = bcinp(self.length,self.time_end,self.bc_pts,self.dt,scl="False")[1]
        return bc_data_r
    
    def bc_scl_r(self):

        bc_data_scaled = self.scale2(self.get_bc_r(), self.x_c, self.t_c)
        return bc_data_scaled
    
    def bc_scl_l(self):
        bc_data_scaled = self.scale2(self.get_bc_l(), self.x_c, self.t_c)
        return bc_data_scaled
    

# scale the inputs as per pde





# tensor the inputs and targets



# tensor the constants from the settings and scale them 


