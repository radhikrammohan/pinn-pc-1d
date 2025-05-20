import sys

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import skopt
from distutils.version import LooseVersion
import csv


def quasirandom(n_samples, sampler, x_min,x_max, t_min, t_max):
    # module to create quasi-random sampling for pde input data
    space = [(x_min, x_max), (t_min, t_max)]
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs(
            lhs_type="centered", criterion="maximin", iterations=1000
        )
    elif sampler == "Halton":
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampler == "Hammersley":
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    elif sampler == "Sobol":
        # Remove the first point [0, 0, ...] and the second point [0.5, 0.5, ...], which
        # are too special and may cause some error.
        if LooseVersion(skopt.__version__) < LooseVersion("0.9"):
            sampler = skopt.sampler.Sobol(min_skip=2, max_skip=2, randomize=False)
        else:
            sampler = skopt.sampler.Sobol(skip=0, randomize=False)
            return np.array(
                sampler.generate(space, n_samples + 2)[2:]
            )
    return np.array(sampler.generate(space, n_samples))

def unidata(x_min, x_max, t_min, t_max, n_samples, sampler):
    # module to create uniform sampling and random sampling for pde input data
    if sampler == "random":
        x = np.random.uniform(x_min, x_max, n_samples)
        t = np.random.uniform(t_min, t_max, n_samples)
        inp = np.column_stack((x, t))
    elif sampler == "uniform":
        x = np.linspace(x_min, x_max, n_samples)
        t = np.linspace(t_min, t_max, n_samples)
        inp = np.column_stack((x, t))
    return inp





def pdeinp(x_min, x_max, t_min, t_max, n_samples, sampler):
     
    # module to create PDE inputs with various sampling strategies
    # define a sampling strategy
    if sampler == "random":
        inp_pde = unidata(x_min, x_max, t_min, t_max, n_samples, sampler)
    elif sampler == "uniform":
        inp_pde = unidata(x_min, x_max, t_min, t_max, n_samples, sampler)
    elif sampler == "LHS":
        inp_pde = quasirandom(n_samples, "LHS", x_min, x_max, t_min, t_max)
    elif sampler == "Halton":
        inp_pde = quasirandom(n_samples, "Halton", x_min, x_max, t_min, t_max)
    elif sampler == "Hammersley":
        inp_pde = quasirandom(n_samples, "Hammersley", x_min, x_max, t_min, t_max)
    elif sampler == "Sobol":
        inp_pde = quasirandom(n_samples, "Sobol", x_min, x_max, t_min, t_max)
    else:
        raise ValueError("Invalid sampler specified. Choose from 'random', 'uniform', 'LHS', 'Halton', 'Hammersley', 'Sobol'.")
    # if scl=="True":
    #     inp_pde[:,0] = scaler(inp_pde[:,0], x_min, x_max)
    #     inp_pde[:,1] = scaler(inp_pde[:,1], t_min, t_max)
    print("The number of points in the PDE input is", len(inp_pde))
    return inp_pde

    #sample the data between input and out

    #meshgrid the same

    #flatten the meshgrid and return the output

def icinp(length, icpts,scl="True"):
    # module to create initial condition inputs
    x = np.linspace(0, length, icpts)
    t= np.zeros(len(x))
    print("The number of points in the initial condition is", len(x))
    if scl == "True":
        x = scaler(x, 0, length)
        
    inp_ic = np.column_stack((x, t))
    return inp_ic

def bcinp(length, time_end, bcpts, delt, scl="True"):
    # module to create boundary condition inputs
    x_l = np.zeros(bcpts)
    x_r = np.ones(bcpts)*length

    t = np.linspace(0+delt, time_end, bcpts)
    print("The number of points in the left boundary condition is", len(x_l))
    print("The number of points in the right boundary condition is", len(x_r))

    if scl == "True":
        x_l = scaler(x_l, 0, length)
        x_r = scaler(x_r, 0, length)
        t = scaler(t, 0, time_end)
    inp_bcl = np.column_stack((x_l, t))
    inp_bcr = np.column_stack((x_r, t))
    return inp_bcl, inp_bcr
    

def scaler(data, min, max):
    # Scale the data between 0 and 1
    scaled_data = (data-min)/(max-min)
    return scaled_data

def invscaler(data, min, max):
    # Inverse scaling to bring the data back to original scale
    invsc_data = data*(max-min) + min
    return invsc_data