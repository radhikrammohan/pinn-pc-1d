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
        self.temperature = temperature
        self.niyama_critical = niyama_critical
        self.material = {
            'kappa': 1.0,
            'rho': 1.0,
            'c': 1.0,
            'L': 1.0,
            'T_melt': 1.0,
            'T_sol': 1.0,
            'T_eutectic': 1.0,
            'T_solidus': 1.0,
            'T_liquidus': 1.0,
        }
        
    def compute_dimensionless_niyama(lambda_sdas, delta_P_cr, mu_liq, rho_liq, rho_sol, delta_Tf):
        """
        Compute the dimensionless Niyama criterion Ny*

        Parameters:
            lambda_sdas : float or np.ndarray
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
        beta = (rho_sol - rho_liq) / rho_liq
        num = lambda_sdas * delta_P_cr * g
        denom = (mu_liq * beta * delta_Tf * T_dot)**(1/2)--
        Ny_star = num/denom 
        return Ny_star
    
    def plot_niyama(self):
        """
        Plots the Niyama values with highlighted critical values.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.imshow(self.niyama, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Niyama Value')
        plt.contour(self.niyama, levels=[self.niyama_critical], colors='blue')
        plt.title('Niyama Values with Critical Contour')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()