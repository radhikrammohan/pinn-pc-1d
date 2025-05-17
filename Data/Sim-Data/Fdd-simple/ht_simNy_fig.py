import sys
from distutils.version import LooseVersion
import numpy as np
import matplotlib.pyplot as plt
import skopt


class HT_sim():
    
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)
            
        # Material properties
        # self.rho = 2300.0     # Density of AL380 (kg/m^3)
        # self.rho_l = 2460.0   # Density of AL380 (kg/m^3)
        # self.rho_s = 2710.0   # Density of AL380 (kg/m^3)
        self.rho_m = (self.rho_l + self.rho_s )/2       # Desnity in mushy zone is taken as average of liquid and solid density

        # self.k = 104.0                       # W/m-K
        # self.k_l = self.k                       # W/m-K
        # self.k_s = 96.2                    # W/m-K
        self.k_m =  (self.k_l+self.k_s)/2                     # W/m-K
        # self.k_mo = 41.5

        # self.cp = 1245.3                      # Specific heat of aluminum (J/kg-K)
        # self.cp_l = self.cp                      # Specific heat of aluminum (J/kg-K)
        # self.cp_s = 963.0                 # Specific heat of aluminum (J/kg-K)
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
        self.temp_array = self.datagen()[0]
        
        self.C_lambda = 40.0e-06                                                                 # C Lambda for Niyama number calculation
        self.del_Pcr = 1.01e5                                                                   # Critical Pressure difference
        self.dyn_visc = 1.2e-3  
        self.current_time = self.datagen()[2]
        self.Niyama_array = self.Ny_calc(self.temp_array)[0]
        self.Lowest_Niyama = self.Ny_calc(self.temp_array)[1]
        self.Avg_Niyama = self.Ny_calc(self.temp_array)[2]
        self.indices = self.Ny_calc(self.temp_array)[3]
        
    def dx_calc(self,length, num_points):
        dx = length / (num_points - 1)
        return dx
    
    def dt_calc(self,dx, alpha_l, alpha_s, alpha_m):
        maxi = max(alpha_s,alpha_l,alpha_m)
        dt = abs(0.5*((dx**2) /maxi))
        return dt
    def step_coeff_calc(self,dt, dx):
        step_coeff = dt / (dx ** 2)
        return step_coeff
    
    def kramp(self,temp,v1,v2,T_L,T_s):   # Function to calculate thermal conductivity in Mushy Zone
            slope = (v1-v2)/(T_L-T_s)
            if temp > T_L:
                k_m = self.k_l
            elif temp < T_s:
                k_m = self.k_s
            else:
                k_m = self.k_s + slope*(temp-T_s)
            return k_m

    def cp_ramp(self,temp,v1,v2,T_L,T_s):    # Function to calculate specific heat capacity in Mushy Zone
        slope = (v1-v2)/(T_L-T_s)
        if temp > T_L:
            cp_m = self.cp_l
        elif temp < T_s:
            cp_m = self.cp_s
        else:
            cp_m = self.cp_s + slope*(temp-T_s)
        return cp_m

    def rho_ramp(self,temp,v1,v2,T_L,T_s):   # Function to calculate density in Mushy Zone
        slope = (v1-v2)/(T_L-T_s)
        if temp > T_L:
            rho_m = self.rho_l
        elif temp < T_s:
            rho_m = self.rho_s
        else:
            rho_m = self.rho_s + slope*(temp-T_s)
        return rho_m     

       
    def datagen(self):
        

        
        current_time = self.dt  
        time_end = 1 
        
        # Initial temperature and phase fields
        temperature = np.full(self.num_points, self.temp_init)             # Initial temperature field 
        phase = np.zeros(self.num_points)*1.0                        # Initial phase field

        # Set boundary conditions
        # temperature[-1] = 723.0 #(40 C)
        phase[-1] = 1.0                                        # Set right boundary condition for phase                  

        # temperature[0] = 723.0 #(40 C)
        phase[0] = 1.0                                             # Set left boundary condition for phase
        

        # Store initial state in history
        temperature_history = [temperature.copy()]                    # Temperature history
        phi_history = [phase.copy()]                                  # Phase history
        temp_initf = temperature.copy()                                   # Additional temperature field for updating
        
        t_surr = self.t_surr                                              # Surrounding temperature             
        dm = 60.0e-3                                                  # thickness of the mold in mm
        
        
        r_m = self.k_mo / dm                                                # Thermal Resistance of the mold
        step_coefff = self.step_coeff_calc(self.dt,self.dx)
        # midpoint_index = num_points // 2
        alpha_l = self.k_l/(self.rho_l*self.cp_l)
        alpha_s = self.k_s/(self.rho_s*self.cp_s)
        # midpoint_temperature_history = [temperature[midpoint_index]]
        
        while current_time < time_end:  # time loop
            htc_l = self.htc_l
            htc_r = self.htc_r
            q1 = htc_l * (temp_initf[0] - t_surr)                     # Heat flux at the left boundary
            alpha_l = self.k_l/(self.rho_l*self.cp_l)
            
            temperature[0] = temp_initf[0] + \
                            (alpha_l * step_coefff * \
                            ((2.0*temp_initf[1]) - \
                            (2.0 * temp_initf[0])-(2.0*self.dx*(q1))))  # Update left boundary condition temperature
            
            q2 = htc_r *(temp_initf[-1]-t_surr)                          # Heat flux at the right boundary
            temperature[-1] = temp_initf[-1] + \
                            (alpha_l * step_coefff \
                                * ((2.0*temp_initf[-2]) - \
                                (2.0 * temp_initf[-1])-(2.0*self.dx*(q2))))  # Update right boundary condition temperature               
            
            for n in range(1,self.num_points-1):              # space loop, adjusted range
                    
                if temperature[n] >= self.T_L:
                    temperature[n] += ((alpha_l * step_coefff) * (temp_initf[n+1] - (2.0 * temp_initf[n]) + temp_initf[n-1]))
                    phase[n] = 0
                            
                elif self.T_S < temperature[n] < self.T_L:
                
                    k_m = self.kramp(temperature[n],self.k_l,self.k_s,self.T_L,self.T_S)
                    cp_m = self.cp_ramp(temperature[n],self.cp_l,self.cp_s,self.T_L,self.T_S)
                    rho_m = self.rho_ramp(temperature[n],self.rho_l,self.rho_s,self.T_L,self.T_S)
                    m_eff =(k_m/(rho_m*(cp_m + (self.L_fusion/(self.T_L-self.T_S)))))
                    
                    temperature[n] +=  ((m_eff * step_coefff)* (temp_initf[n+1] - (2.0 * temp_initf[n]) + temp_initf[n-1]))
                    phase[n] = (self.T_L - temperature[n]) / (self.T_L - self.T_S) 
                    
                elif temperature[n] <= self.T_S:
                    temperature[n] +=  ((alpha_s * step_coefff) * (temp_initf[n+1] - (2.0 * temp_initf[n])+ temp_initf[n-1]))
                    phase[n] = 1
                
                else:
                    print("ERROR: should not be here")
            
                
            current_time = current_time + self.dt                                         # Update current time
            time_end = time_end +self.dt                                                 # Update end time
            temperature = temperature.copy()                                        # Update temperature field
            phase = phase.copy()                                                        # Update phase field
            temp_initf = temperature.copy()                                          # Update additional temperature field
            temperature_history.append(temperature.copy())                           # Store temperature field in history
            phi_history.append(phase.copy())                                             # Store phase field in history
            # midpoint_temperature_history.append(temperature[midpoint_index])        
            if np.all(phase == 1):
                # print("Simulation complete @ time: ", current_time)
                break
            
            # Check the new shape after transposing
        
        temperature_history_1 = np.array(temperature_history)                       # Convert temperature history to numpy array
        phi_history_1 = np.array(phi_history)                                      # Convert phase history to numpy array
        
        return temperature_history_1, phi_history_1,current_time
    
        self.temp_array = self.datagen()[0]
    
    def Ny_calc(self, temp_array):
        
        
        temp_hist_l = temp_array[:,1:-1]
        
        print(temp_hist_l.shape)
        t_dim,x_dim  = temp_hist_l.shape
        # Niyama Calcualtion

        # print(temperature_history_1.shape)
        # Gradient Calculation

        grad_t_x = np.absolute(np.gradient(temp_array, self.dx, axis=1))     # Gradient of temperature with respect to space
        
        # print(grad_t_x[100,:])
        grad_t_t = np.absolute(np.gradient(temp_array,self.dt,axis=0))       # Gradient of temperature with respect to time
        
        # print(grad_t_t[100,:])
        sq_grad_t_t = np.square(grad_t_t)                                         # Square of the gradient of temperature with respect to space
        Ny = np.divide(grad_t_x, sq_grad_t_t, out=np.zeros_like(grad_t_x, dtype=float), where=sq_grad_t_t!=0)         # Niyama number
        # print(Ny)


                                                                         # Dynamic Viscosity
        beta = (self.rho_s - self.rho_l)/ self.rho_l                                                     # Beta    
        # print(beta)
        del_Tf = self.T_L - self.T_S                                                               # Delta T
        # print(del_Tf)
        k1a=(self.dyn_visc*beta*del_Tf)                                                      
        k1 = (self.del_Pcr/k1a)**(1/2)
        # print(k1)
        num_steps = temp_hist_l.shape[0]-1                                # Number of time steps
        # print(num_steps)

        # k2 = np.divide(grad_t_x, grad_t_t_power, out=np.zeros_like(grad_t_x, dtype=float), where=grad_t_t_power!=0)
        k2 = np.zeros((num_steps+1,self.num_points))
        k3 = np.zeros((num_steps+1,self.num_points))
        for i in range(num_steps+1):
            for j in range(self.num_points):
                if grad_t_x[i,j] == 0:
                    k2[i,j] = 0
                    k3[i,j] = 0
                if grad_t_t[i,j]== 0:
                    k2[i,j] = 0
                    k3[i,j] = 0
                else:
                    k2[i,j] = ((grad_t_x[i,j]))/ (((grad_t_t[i,j]))**(5/6))
                    k3[i,j] = (grad_t_x[i,j])/ ((grad_t_t[i,j])**(1/2))
            
        # k2 = grad_t_x/((grad_t_t)**(5/6))
        # print(k2)
        Ny_s= k3
        Dim_ny = self.C_lambda * k1 * k2
        # print(Dim_ny)

        # print(grad_t_t[:, 50])
        # plot = plt.figure(figsize=(10, 6))
        # plt.plot(time_ss, grad_t_x[:, 50], label='Niyama Number at x = 7.5mm')
        # plt.xlabel('Time(s)')
        # plt.ylabel('Niyama Number')
        # plt.title('Niyama Number Distribution Over Time at x = 7.5mm')
        # plt.legend()
        # plt.show()**
        
        current_time = self.current_time
        Ny_time = 0.90*current_time                                     # Time at which Niyama number is calculated 

        Ny_index = int(Ny_time/self.dt)                                      # Index of the time at which Niyama number is calculated
        Cr_Ny = np.min(Dim_ny[Ny_index, :])
        Cr_Nys = np.min(Ny_s[Ny_index,:])                                # Minimum Niyama number at the time of interest
        
        indices =[]
        threshold = self.T_S + 0.1*(self.T_L-self.T_S)
        tolerance = 1.0
        # print(threshold)

        for i in range (t_dim):
            for j in range(x_dim):
                if np.absolute(temp_hist_l[i,j]- threshold) < tolerance:
                    indices.append((i,j))

        # print(indices)

        

        # print(Dim_ny.shape)
        Niyama_pct = [Dim_ny[i,j] for i,j in indices]
        Niyama_array = np.array(Niyama_pct)
        Lowest_Niyama = np.min(Niyama_array)
        Avg_Niyama = np.mean(Niyama_array)

        return Dim_ny, Lowest_Niyama, Avg_Niyama, indices


        # # Check the new shape after transposing
        # print("Transposed Temperature History Shape:", temperature_history.shape)
        # print("Transposed Phi History Shape:", phi_history.shape)
    def Ny_plot(self):
        
        temp_array = self.temp_array
        # Create a meshgrid for space and time coordinates
        space_coord, time_coord = np.meshgrid(np.arange(temp_array.shape[1]), np.arange(temp_array.shape[0]))

        time_coord = time_coord * self.dt 

        hlt_t, hlt_x = zip(*self.indices)
        real_t = []

        for index in self.indices:
            real_t.append(time_coord[index[0],index[1]])

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 8))

        # Plot the temperature history on the left subplot
        im1 = ax1.pcolormesh(space_coord, time_coord, temp_array, cmap='viridis')
        ax1.set_xlabel('Space Coordinate', fontname='Times New Roman', fontsize=16)
        ax1.set_ylabel('Time',fontname='Times New Roman', fontsize=16)
        ax1.set_title('Temperature Variation Over Time',fontname='Times New Roman', fontsize=20)
        fig.colorbar(im1, ax=ax1, label='Temperature')

        # # Plot the phase history on the right subplot
        # im2 = ax2.pcolormesh(space_coord, time_coord, phi_history_1, cmap='viridis')
        # ax2.set_xlabel('Space Coordinate', fontname='Times New Roman', fontsize=18)
        # ax2.set_ylabel('Time',fontname='Times New Roman', fontsize=16)
        # ax2.set_title('Phase Variation Over Time',fontname='Times New Roman', fontsize=20)
        # fig.colorbar(im2, ax=ax2, label='Phase')


        #plot the main
        # fig, ax = plt.subplots(figsize=(14, 6))

        im2 = ax2.pcolormesh(space_coord, time_coord, self.Niyama_array, cmap='viridis')
        im2 = ax2.scatter(hlt_x, real_t, color='r', s=1, zorder=5, label='Highlighted Points')  # s=100 sets the marker size, zorder=5 puts the points on top
        ax2.set_xlabel('Space Coordinate')
        ax2.set_ylabel('Time')
        ax2.set_title('Niyama Variation Over Time')
        fig.colorbar(im2, ax= ax2, label='Dimensionless Niyama Number')



        plt.tight_layout()
        plt.show()




        # print(f'Lowest Niyama:{Lowest_Niyama}, rho_l:{rho_l}, rho_s:{rho_s}, k_l:{k_l}, k_s:{k_s}, cp_l:{cp_l}, cp_s:{cp_s}, t_surr:{t_surr}, L_fusion:{L_fusion}, temp_init:{temp_init},self.:{self.},htc_r:{htc_r}')
        



