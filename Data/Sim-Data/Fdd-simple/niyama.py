import matplotlib.pyplot as plt
import numpy as np

class Niyama:
    
    """
    Niyama class for calculating Niyama values based on
    a given temperature array.
    
    Attributes
    ----------
    temperature : numpy.ndarray
        A 2D array representing the temperature distribution.
    niyama : numpy.ndarray
        A 2D array representing the Niyama values.
    niyama_critical : float
    material variables
        A dictionary containing material properties.
    
    Methods
    -------
    calculate_niyama()
        Calculates the Niyama values based on the temperature array.
    get_niyama()
        Returns the Niyama values.
    get_niyama_critical()
        Returns the critical Niyama value.
    plot_niyama()
        Plots the Niyama values with highlihgted niyama critical values.
    """
    
    def __init__(self, temperature, niyama_critical=0.5):
        """
        Initializes the Niyama class with a temperature array and
        a critical Niyama value.
        
        Parameters
        ----------
        temperature : numpy.ndarray
            A 2D array representing the temperature distribution.
        niyama_critical : float, optional
            The critical Niyama value (default is 0.5).
        """
        self.length = 15.0e-3
        self.time_end = 5.0
        self.temperature = temperature
        self.niyama_critical = niyama_critical
        self.c_lambda = 40.0e-06 # in meter
        self.delta_P_cr = 1.01e5 # in bar
        self.mu_liq = 1.58e-3   
        self.rho_liq = 2369.0
        self.rho_sol = 2400.0
        self.delta_Tf = 56.0
        self.dx = self.length / temperature.shape[1]  # spatial step size - length divided by number of x points
        self.dt = self.time_end / (temperature.shape[0] - 1) # Subtract 1 since n intervals require n+1 points (e.g. 0s to 5s with 0.1s steps needs 51 points)
        self.num_steps = temperature.shape[0]-1
        self.num_points = temperature.shape[1]
        self.g = np.absolute(np.gradient(temperature, self.dx,
                             axis=0))
        self.T_dot =np.absolute(np.gradient(temperature, self.dt,
                                 axis=1))
    def compute_dimensionless_niyama(self):
        """
        Compute the dimensionless Niyama criterion Ny*

        Parameters:
            c_lambda : float or np.ndarray
                Secondary dendrite arm spacing (in meters)
            delta_P_cr : float
                Critical pressure drop (Pa)
            mu_liq : float
                Dynamic viscosity of liquid metal (Pa·s)
            rho_liq : float
                Density of liquid phase (kg/m³)
            rho_sol : float
                Density of solid phase (kg/m³)
            delta_Tf : float
                Freezing range (T_l - T_s) in K or °C

        Returns:
            Ny_star : float or np.ndarray
                Dimensionless Niyama criterion
        """
        
        beta = (self.rho_sol -self.rho_liq)/ self.rho_sol
        k1a = (self.mu_liq * beta * self.delta_Tf)
        k1 = (self.delta_P_cr /k1a) **(1/2)
        
        k2 = np.zeros((self.num_steps+1,self.num_points))
        k3 = np.zeros((self.num_steps+1,self.num_points))
        
        for i in range(self.num_steps+1):
            for j in range(self.num_points):
                if self.g[i,j] == 0:
                    k2[i,j] = 0
                    k3[i,j] = 0
                if self.T_dot[i,j] == 0:
                    k2[i,j] = 0
                    k3[i,j] = 0
                else:
                    k2[i,j] = ((self.g[i,j]))/ (((self.T_dot[i,j]))**(5/6))
                    k3[i,j] = (self.g[i,j])/ ((self.T_dot[i,j])**(1/2))
        Ny_star = k3
        Dim_ny = self.c_lambda * k1 * k2
        return Ny_star, Dim_ny 
        
    # def plot_niyama(self):
    #     """
    #     Plots the Niyama values with highlighted critical values.
    #     """
    #     Ny_time = 0.90*self.time_end                                    # Time at which Niyama number is calculated 

    #     Ny_index = int(Ny_time/self.dt)                                      # Index of the time at which Niyama number is calculated
        
    #     Cr_Ny = np.min(Dim_ny[Ny_index, :])
    #     Cr_Nys = np.min(Ny_s[Ny_index,:])                                # Minimum Niyama number at the time of interest
        
    #     indices =[]
    #     threshold = self.T_S + 0.1*(T_L-T_S)
    #     tolerance = 1.0
    #     # print(threshold)

    #     for i in range (t_dim):
    #         for j in range(x_dim):
    #             if np.absolute(temp_hist_l[i,j]- threshold) < tolerance:
    #                 indices.append((i,j))
    #         space_coord, time_coord = np.meshgrid(np.arange(self.temperature.shape[1]), np.arange(self.temeprature.shape[0]))

    #         time_coord = time_coord * self.dt 

    #         hlt_t, hlt_x = zip(*indices)
    #         real_t = []

    #         for index in indices:
    #             real_t.append(time_coord[index[0],index[1]])

    #     # Create a figure with two subplots
    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 8))

    #     # Plot the temperature history on the left subplot
    #     im1 = ax1.pcolormesh(space_coord, time_coord, temp_hist_l, cmap='viridis')
    #     ax1.set_xlabel('Space Coordinate', fontname='Times New Roman', fontsize=16)
    #     ax1.set_ylabel('Time',fontname='Times New Roman', fontsize=16)
    #     ax1.set_title('Temperature Variation Over Time',fontname='Times New Roman', fontsize=20)
    #     fig.colorbar(im1, ax=ax1, label='Temperature')

    #     # Plot the phase history on the right subplot
    #     im2 = ax2.pcolormesh(space_coord, time_coord, phi_history_1, cmap='viridis')
    #     ax2.set_xlabel('Space Coordinate', fontname='Times New Roman', fontsize=18)
    #     ax2.set_ylabel('Time',fontname='Times New Roman', fontsize=16)
    #     ax2.set_title('Phase Variation Over Time',fontname='Times New Roman', fontsize=20)
    #     fig.colorbar(im2, ax=ax2, label='Phase')


    #     #plot the main
    #     # fig, ax = plt.subplots(figsize=(14, 6))

    #     im3 = ax3.pcolormesh(space_coord, time_coord, Dim_ny[:,1:-1], cmap='viridis')
    #     im3 = ax3.scatter(hlt_x, real_t, color='r', s=1, zorder=5, label='Highlighted Points')  # s=100 sets the marker size, zorder=5 puts the points on top
    #     ax3.set_xlabel('Space Coordinate')
    #     ax3.set_ylabel('Time')
    #     ax3.set_title('Niyama Variation Over Time')
    #     fig.colorbar(im3, ax= ax3, label='Dimesnionless Niyama Number')



    #     plt.tight_layout()
    #     plt.show()
        
