'''

Module: 1D Heat Transfer Simulation
-----------------------------------
This module provides a class and supporting functions to simulate heat transfer in a 1D rod along with mushy zone calculations.
It supports solving the heat conduction equation with given material properties, initial/boundary conditions,
and generates the temperature profile over time.

Main Components:
- HT_sim: Class for 1D heat transfer simulation
- fdd: Generates a grid of spatial-temporal points for the finite difference solution
- pdeinp: Sampling strategies for PDE training inputs
- icinp: Generates input points for initial conditions
- bcinp: Generates input points for boundary conditions
- scaler / invscaler: Utility functions for normalization

Dependencies:
- numpy, matplotlib, skopt

Example usage:
    sim = HT_sim(length=0.06, time_end=10, num_points=100, t_surr=300, temp_init=900)
    result = sim.datagen()
    sim.plot_temp(idx=50)

'''
# import sys
# from distutils.version import LooseVersion

import numpy as np
import matplotlib.pyplot as plt
import skopt

from joblib import load

# Geometry

class HT_sim():
    # class to simulate heat transfer in a rod
    # the class has the following attributes like material properties, 
    # length of the rod, time of simulation, number of points, initial temperature, and surrounding temperature
    # the class has the following methods like dx_calc, dt_calc, cflcheck, step_coeff_calc, datagen, plot_temp
    

    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)

        
        self.rho_m = (self.rho_l + self.rho_s )/2       # Desnity in mushy zone is taken as average of liquid and solid density

      
        self.k_m =  (self.k_l+self.k_s)/2                     # W/m-K
        
        self.cp_m =  (self.cp_l+self.cp_s)/2                 # Specific heat of mushy zone is taken as average of liquid and solid specific heat

        self.alpha_l = self.k_l / (self.rho_l * self.cp_l)
        self.alpha_s = self.k_s / (self.rho_s*self.cp_s)
        self.alpha_m = self.k_m / (self.rho_m * self.cp_m)          #`Thermal diffusivity in mushy zone is taken as average of liquid and solid thermal diffusivity`
        # print(self.alpha_m, self.alpha_l, self.alpha_s)
        # self.L_fusion = 389.0e3               # J/kg  # Latent heat of fusion of aluminum
        # self.T_L = 574.4 +273.0                       #  K -Liquidus Temperature (615 c) AL 380
        # self.T_S = 497.3 +273.0                     # K- Solidus Temperature (550 C)
        self.m_eff =(self.k_m/(self.rho_m*(self.cp_m + (self.L_fusion/(self.T_L-self.T_S)))))

        # sim field generation
        self.tempfield = np.full(self.num_points, self.temp_init)            # Initial temperature of the rod with ghost points at both ends
                         # Initial temperature of the rod

                                # Index of the midpoint
                  # List to store temperature at midpoint over time
                                                                   # die thickness in m
        # Calculate dx,dt, and step_coeff
        self.dx = self.dx_calc(self.length, self.num_points)
        print("The spatial step is", self.dx)
        self.dt = self.dt_calc(self.dx, self.alpha_l, self.alpha_s, self.alpha_m)
        print("The time step is", self.dt)
        self.step_coeff = self.step_coeff_calc(self.dt, self.dx)
        self.num_steps = round(self.time_end/self.dt)

        
        self.model_k_path = config["model_k_path"]
        self.model_cp_path = config["model_cp_path"]
        
        self.model_k = load(self.model_k_path)
        self.model_cp = load(self.model_cp_path)

    def dx_calc(self,length, num_points):
        dx = length / (num_points - 1)
        return dx
    
    def dt_calc(self,dx, alpha_l, alpha_s, alpha_m):
        maxi = max(alpha_s,alpha_l,alpha_m)
        dt = abs(0.5*((dx**2) /maxi))
        return dt
    def cflcheck(self,dx, alpha_l, alpha_s, alpha_m):
        cfl = 0.5 *(dx**2/max(alpha_l,alpha_s,alpha_m))
        if cfl > 1:
            print("CFL condition not satisfied")
            sys.exit()
        else:
            print("CFL condition satisfied")
        
        return cfl
    def step_coeff_calc(self,dt, dx):
        step_coeff = dt / (dx ** 2)
        return step_coeff

    def latent_heat(self,t):
        """
        Calculate the latent heat of fusion for a given temperature or temperatures
        based on Zhang paper Data. Handles scalar, list, or numpy array inputs.
        """
        # Convert input to a numpy array for vectorized operations
        t = np.array(t, dtype=float)

        # Initialize latent heat with default value (e.g., 0 for out-of-range values)
        lf = np.zeros_like(t)

        # Apply conditions for each range
        lf[(t >= (533.0 +273.0)) & (t < (563.6+273))] = 36.18
        lf[(t >= (563.6+273.0)) & (t < (567.2+273))] = 170.36
        lf[(t >= (567.2+273.0)) & (t < (588.2+273))] = 48.74
        lf[(t >= (588.2+273.0)) & (t < (610.7+273))] = 91.20
        lf[(t >= (610.7+273.0)) & (t < (613.2+273))] = 51.02

        # Return scalar if input was scalar, otherwise return array
        return lf.item() if lf.size == 1 else lf

    def rho_ramp(self,temp,v1,v2,T_L,T_S):                                       # Function to calculate density in Mushy Zone
        slope = (v1-v2)/(T_L-T_S)
        if temp > T_L:
            rho_m = v1
        elif temp < T_S:
            rho_m = v2
        else:
            rho_m = v2 + slope*(temp-T_S)
        return rho_m
    
    def datagen(self):

        tempfield = self.tempfield.copy() 
       
        temp_int = self.tempfield.copy()
        self.temphist = [tempfield.copy()]
        
        for m in range(1, self.num_steps+1):                                                                            # time loop
            
            
            
            # print(f"q1 is {q1}")
            tempfield[0] = self.die_temp_l
                         
            tempfield[-1] = self.die_temp_r

            for n in range(1,self.num_points-1):              # space loop, adjusted range
                
                if tempfield[n] > self.T_L:  # Liquid phase
                    tempfield[n] += ((self.alpha_l * self.step_coeff) * (temp_int[n+1] \
                        - (2.0 * temp_int[n]) + temp_int[n-1]))
                    
                elif self.T_L >= tempfield[n] > self.T_S:  # Mushy phase
                    
                    k_m = self.model_k.predict([[tempfield[n]]]) # include the model for k from Zhang
                    cp_m = self.model_cp.predict([[tempfield[n]]]) # include the model for cp from Zhang
                    rho_m = self.rho_ramp(tempfield[n], self.rho_l, self.rho_s, self.T_L, self.T_S) #inlcude the model for rho from Zhang
                    self.alpha_m = k_m / ((rho_m * (cp_m)) - self.latent_heat(tempfield[n]))# Figureout how the eqn cameup
                    
                    tempfield[n] += ((self.alpha_m * self.step_coeff) * (temp_int[n+1] \
                        - (2.0 * temp_int[n]) + temp_int[n-1]))
                
                elif tempfield[n] <= self.T_S:  # Solid phase
                    tempfield[n] += ((self.alpha_s * self.step_coeff) * (temp_int[n+1] \
                        - (2.0 * temp_int[n]) + temp_int[n-1]))
                    
                else:  # Invalid temperature range
                    raise ValueError(f"Temperature {tempfield[n]} at index {n} is out of bounds.")
                                                                      # Update temperature
            temp_int = tempfield.copy()                                                                  # Update last time step temperature
            self.temphist.append(tempfield.copy())                                                  # Append the temperature history to add ghost points
                                            # Store midpoint temperature
        
        self.temp_history_1 = np.array(self.temphist)
        
        return self.temp_history_1
     
    def space_time(self):
        # Create a meshgrid for space and time
        x = np.linspace(0, self.length, self.num_points)
        t = np.linspace(0, self.time_end, self.num_steps+1)
        X, T = np.meshgrid(x, t)
        return X, T 
    
    def plot_temp(self,idx):
        # Plot the temperature distribution over time at the midpoint
        time_ss= np.linspace(0, self.time_end, self.num_steps+1)
        dx = self.dx
        plt.figure(figsize=(10, 6))
        plt.plot(time_ss, self.temp_history_1[:,idx], label='Midpoint Temperature')
        plt.axhline(y=self.T_L, color='r', linestyle='--', label='Liquidus Temperature')
        plt.axhline(y=self.T_S, color='g', linestyle='--', label='Solidus Temperature')
        plt.xlabel('Time(s)')
        plt.ylabel('Temperature (K)')
        plt.title(f'Temperature Distribution Over Time at x = {idx*dx*1000:.2f} mm') 
        plt.legend()
        plt.show()

 
                                                             # Update temperature

def fdd(length, time_end, num_points, num_steps, scl="True"):
    # module to create finite difference data
    x = np.linspace(0, length, num_points)
   
    t = np.linspace(0, time_end, num_steps)
    X, T = np.meshgrid(x, t)
    x = X.flatten()
    t = T.flatten()
    # print(x)
    # if scl == "True":
    #     x = scaler(x, 0, length)
    #     t = scaler(t, 0, time_end) 
    inp_fdd = np.column_stack((x, t))
    return inp_fdd

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





def pdeinp(x_min, x_max, t_min, t_max, n_samples, sampler, scl="True"):
     
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
        raise ValueError("Invalid sampler specified. Choose from 'random',\
            'uniform', 'LHS', 'Halton', 'Hammersley', 'Sobol'.")
    if scl=="True":
        print("scaling initated")
        inp_pde[:,0] = scaler(inp_pde[:,0], x_min, x_max)
        inp_pde[:,1] = scaler(inp_pde[:,1], t_min, t_max)
    else:
        print("scaling not initiated")
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
    else:
        print("scaling not initiated")
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
    else:
        print("scaling not initiated")
    inp_bcl = np.column_stack((x_l, t))
    inp_bcr = np.column_stack((x_r, t))
    return inp_bcl, inp_bcr
    

def scaler(data, min_d, max_d):
    # Scale the data between 0 and 1
    scaled_data = (data-min_d)/(max_d-min_d)
    return scaled_data

def invscaler(data, min_d, max_d):
    # Inverse scaling to bring the data back to original scale
    invsc_data = data*(max_d-min_d) + min_d
    return invsc_data