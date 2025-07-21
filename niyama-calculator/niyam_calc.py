"""
NiyamaCalc Module
=================

Author: Radhik Rammohan  
Created: July 17, 2025

Description:
------------
This module provides a class to compute the Niyama number field from a temperature field, 
along with key metadata such as the lowest Niyama value and its location.

The Niyama number is used in casting and solidification modeling to predict defects such as 
shrinkage porosity based on thermal gradients and cooling rates.

Inputs:
-------
- temperature_field : np.ndarray
    A 2D array (time x space) representing the temperature evolution at spatial points over time.
- mat_proc_dict : dict
    A dictionary containing the required material and process parameters:
        - dx : float (spatial step)
        - dt : float (time step)
        - rho_l, rho_s : float (liquid and solid densities)
        - T_l, T_s : float (liquidus and solidus temperatures)
        - del_Pcr : float (critical pressure drop)
        - dyn_visc : float (dynamic viscosity)
        - C_lambda : float (material constant)
        - current_time : float (total simulation time)


Outputs:
--------
- niyama_field : np.ndarray
    The 2D Niyama number field (same shape as input temperature field).
- lowest_niyama_value : float
    The lowest Niyama value observed at 90% of simulation time.
- location_of_lowest_niyama_value : int
    The spatial index where the lowest Niyama value was observed.
"""

import numpy as np

class NiyamaCalc:
    def __init__(self, temperature_field, mat_proc_dict):
        self.temperature_field = np.array(temperature_field)  # shape (time, space)
        self.material_process = mat_proc_dict
        

        # Output attributes
        self.niyama_field = None
        self.lowest_niyama_value = None
        self.location_of_lowest_niyama_value = None

        # Extract required parameters from dictionary
        self.dx = mat_proc_dict["dx"] # this needs to calcualted from the field
        self.dt = mat_proc_dict["dt"]
        self.rho_l = mat_proc_dict["rho_l"]
        self.rho_s = mat_proc_dict["rho_s"]
        self.T_l = mat_proc_dict["T_L"]
        self.T_s = mat_proc_dict["T_S"]
        self.del_Pcr = mat_proc_dict["del_Pcr"]
        self.dyn_visc = mat_proc_dict["dyn_visc"]
        self.C_lambda = mat_proc_dict.get("C_lambda")
        self.current_time = mat_proc_dict.get("time_end")
        self.del_Tf = self.T_l - self.T_s
        self.beta = (self.rho_s - self.rho_l) / self.rho_l
        self.k1a = (self.dyn_visc * self.beta * self.del_Tf) 
        self.k1 = np.sqrt(self.del_Pcr / self.k1a)

    def calculate_niyama(self):
        temp = self.temperature_field
        num_time_steps, num_spatial_points = temp.shape
        grad_t_x = np.absolute(np.gradient(temp, self.dx, axis=1))
        grad_t_t = np.absolute(np.gradient(temp, self.dt, axis=0))
        square_grad_t_t = np.square(grad_t_t)
        
        Ny = np.divide(grad_t_x, square_grad_t_t, out=np.zeros_like(grad_t_x,dtype=float), where=square_grad_t_t != 0)

        k2 = np.zeros((num_time_steps+1, num_spatial_points), dtype=float)
        k3 = np.zeros((num_time_steps+1, num_spatial_points), dtype=float)
        
        
        
        for i in range(num_time_steps):
            for j in range(num_spatial_points):
                if grad_t_x[i, j] == 0.0:
                    k2[i, j] = 0.0
                    k3[i, j] = 0.0
                if grad_t_t[i, j] == 0.0:
                    k2[i, j] = 0.0
                    k3[i, j] = 0.0
                else:
                    k2[i,j] = ((grad_t_x[i,j]))/ (((grad_t_t[i,j]))**(5.0/6.0))
                    k3[i,j] = (grad_t_x[i,j])/ ((grad_t_t[i,j])**(1.0/2.0))

        Ny_s = k3
        print(k3)
        Dim_ny = self.C_lambda * self.k1 * k2
        
        

        # Evaluate at 90% of simulation time
        ny_index = int(0.9 * self.current_time / self.dt)
        Cr_ny = np.min(Dim_ny[ny_index, :])
        Cr_nys = np.min(Ny_s[ny_index, :])
        
        indices = []
        threshold = self.T_s + 0.1 * (self.T_l - self.T_s)
        tolerance = 1.0
        
        for i in range(num_time_steps):
            for j in range(num_spatial_points-1):
                if np.absolute(temp[i,j]- threshold) < tolerance:
                    indices.append((i, j))
        
        Niyama_pct = [Dim_ny[i, j] for i, j in indices]
        Niyama_array = np.array(Niyama_pct)
        
        Lowest_ny = np.min(Niyama_array)
        Avg_ny = np.mean(Niyama_array)
        
        self.niyama_field = Dim_ny
        self.lowest_niyama_value = Lowest_ny
        self.location_of_lowest_niyama_value = [np.argmin(Niyama_array),ny_index]
        
        return self.lowest_niyama_value

    def get_results(self):
        return {
            "niyama_field": self.Dim_ny,
            "lowest_niyama_value": self.lowest_niyama_value,
            "location_of_lowest_niyama_value": self.location_of_lowest_niyama_value
        }