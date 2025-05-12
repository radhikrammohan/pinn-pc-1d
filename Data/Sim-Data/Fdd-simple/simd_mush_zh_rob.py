import numpy as np
import matplotlib.pyplot as plt
from joblib import load

class HT_sim:
    def __init__(self, config):
        # --- load user config ---
        for k, v in config.items():
            setattr(self, k, v)

        # material constants (solidus/liquidus temperatures in K)
        self.Ts, self.Tl = config["T_S"], config["T_L"]
        self.Lf            = config["L_fusion"]   # J/kg
        self.h_gap         = 20      # W/m²K (die–casting contact conductance)
        self.T_die_left    = config["die_temp_l"]
        self.T_die_right   = config["die_temp_r"]

        # load surrogates
        self.model_k  = load(self.model_k_path)
        self.model_cp = load(self.model_cp_path)

        # spatial grid
        self.dx = self.length / (self.num_points - 1)
        x = np.linspace(0, self.length, self.num_points)
        # initial enthalpy H = c_p·T (no latent heat yet)
        cp0 = self.model_cp.predict([[self.temp_init]])[0]
        self.H = cp0 * np.full_like(x, self.temp_init)
        self.history = [self.H.copy()]

    def liquid_fraction(self, T):
        """Linearly ramp fraction f from 0 at Ts to 1 at Tl."""
        f = np.clip((T - self.Ts) / (self.Tl - self.Ts), 0, 1)
        return f

    def mixture_properties(self, T):
        """Return ρ, c_p, k for a temperature T array."""
        f = self.liquid_fraction(T)
        # unpack surrogates to 1D arrays
        k_s = float(self.model_k.predict([[self.Ts]])[0])
        k_l = float(self.model_k.predict([[self.Tl]])[0])
        cp_s = float(self.model_cp.predict([[self.Ts]])[0])
        cp_l = float(self.model_cp.predict([[self.Tl]])[0])
        # mix linearly
        k  = f * k_l  + (1 - f) * k_s
        cp = f * cp_l + (1 - f) * cp_s
        ρ  = f * self.rho_l + (1 - f) * self.rho_s
        α  = k / (ρ * cp)
        return ρ, cp, k, α, f

    def step(self):
        """Advance one time step using an explicit enthalpy‐method FDM plus Robin BCs."""
        Hn = self.H.copy()
        # recover T from H: invert H = cp·T + Lf·f(T)
        # here we do a simple Newton iteration per node
        Tn = np.zeros_like(Hn)
        for i, Hval in enumerate(Hn):
            # initial guess
            T = Hval / float(self.model_cp.predict([[self.Ts]])[0])
            for _ in range(5):
                ρ, cp, k, α, f = self.mixture_properties(T)
                H_guess = cp*T + self.Lf * f
                # dH/dT = cp + Lf*(df/dT) = cp + Lf/(Tl-Ts)
                dH_dT  = cp + self.Lf/(self.Tl - self.Ts)
                T -= (H_guess - Hval)/dH_dT
            Tn[i] = T

        ρ, cp, k, α, f = self.mixture_properties(Tn)
        # recompute stable dt if you like
        dt = 0.5 * self.dx**2 / np.max(α)

        # volume update: ∂H/∂t = ∇·(k∇T)
        Hnew = Hn.copy()
        flux = np.zeros_like(Hn)
        # interior points
        for i in range(1, self.num_points-1):
            flux[i] = k[i] * (Tn[i+1] - 2*Tn[i] + Tn[i-1]) / self.dx**2
            Hnew[i] = Hn[i] + dt * flux[i]

        # Robin BC at left end: 
        # -k ∂T/∂x |₀ = h_gap (T₀ - T_die_left)
        # finite‐difference: (T₁ - T₀)/dx
        q_left = -k[0]*(Tn[1]-Tn[0])/self.dx
        Hnew[0] = Hn[0] + dt*( q_left + self.h_gap*(self.T_die_left - Tn[0]) )/self.dx

        # Robin BC at right end:
        q_right = -k[-1]*(Tn[-1]-Tn[-2])/self.dx
        Hnew[-1]= Hn[-1]+ dt*(-q_right + self.h_gap*(self.T_die_right - Tn[-1]))/self.dx

        self.H = Hnew
        self.history.append(self.H.copy())
        return dt

    def run(self):
        """Run all time steps until time_end."""
        t = 0
        while t < self.time_end:
            dt = self.step()
            t += dt
        # at end, convert history→ temperatures for plotting
        self.T_history = []
        for H in self.history:
            # invert enthalpy→T one last time
            # (reuse same Newton routine from step)
            # …
            pass

    def plot_midpoint(self):
        midpoint = self.num_points//2
        Tmid = []
        # for each saved enthalpy snapshot H
        for H in self.history:
            # invert H→T with the same Newton loop you used in step()
            T = np.zeros_like(H)
            for i, Hval in enumerate(H):
                # initial guess = Ts
                tloc = self.Ts
                for _ in range(5):
                    _, cp, _, _, f = self.mixture_properties(tloc)
                    Hguess = cp*tloc + self.Lf*f
                    dH_dT  = cp + self.Lf/(self.Tl - self.Ts)
                    tloc -= (Hguess - Hval)/dH_dT
                T[i] = tloc
            Tmid.append(T[midpoint])
        times = np.linspace(0, self.time_end, len(Tmid))
        plt.plot(times, Tmid, label="Midpoint")
