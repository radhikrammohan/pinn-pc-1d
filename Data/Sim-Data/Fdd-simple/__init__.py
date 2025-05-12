    """
    Finite Difference Discretization (FDD) implementation for simple 1D phase change problems.
    
    This module provides numerical simulation capabilities for phase change processes using
    finite difference methods. It includes functionality for temperature evolution, phase 
    transition tracking, and material property handling based on settings defined in 
    settings.json.

    Modules:
    - settings.json: Contains material properties, simulation parameters and model paths
    - niyama.py: Implements Niyama criterion calculations for solidification analysis
    - solver.py: Core FDD solver implementation for temperature evolution
    - properties.py: Handles temperature-dependent material properties
    - boundary.py: Manages boundary conditions and heat transfer coefficients
    - utils.py: Utility functions for data processing and visualization
    
    The implementation supports:
    - Temperature-dependent material properties
    - Latent heat effects during phase change 
    - Heat transfer coefficient boundary conditions
    - Integration with trained neural network models for property prediction
    - Niyama criterion analysis for casting quality assessment
    """