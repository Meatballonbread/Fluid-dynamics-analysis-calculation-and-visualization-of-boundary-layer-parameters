import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_values(w_m_per_s, nue, lambda_, Pr, x_values):
    results = []

    for x in x_values:
        Re = w_m_per_s * x / nue
        delta_h = 4.64 * Re**(-0.5) * x
        delta_th = 0.976 * Pr**(-1/3) * delta_h
        alpha_lam = 0.332 * Pr**0.33 * Re**0.5 * lambda_ / x

        results.append({
            'x': x,
            'Re': Re,
            'delta_h': delta_h,
            'delta_th': delta_th,
            'alpha_lam': alpha_lam
        })
    
    return results

def plot_values(w_m_per_s, nue, lambda_, Pr, x_values):
    calculated_values = calculate_values(w_m_per_s, nue, lambda_, Pr, x_values)
    df = pd.DataFrame(calculated_values)

    plt.figure(figsize=(14, 10))

    # Reynolds Number (Re) vs x
    plt.subplot(2, 2, 1)
    plt.plot(df['x'], df['Re'])
    plt.title(f'Reynolds Number (Re) vs x (w = {w_m_per_s * 3.6} km/h)')
    plt.xlabel('x')
    plt.ylabel('Re')

    # Hydrodynamic Boundary Layer Thickness (delta_h) vs x
    plt.subplot(2, 2, 2)
    plt.plot(df['x'], df['delta_h'])
    plt.title('Hydrodynamic Boundary Layer Thickness (delta_h) vs x')
    plt.xlabel('x')
    plt.ylabel('delta_h')

    # Thermal Boundary Layer Thickness (delta_th) vs x
    plt.subplot(2, 2, 3)
    plt.plot(df['x'], df['delta_th'])
    plt.title('Thermal Boundary Layer Thickness (delta_th) vs x')
    plt.xlabel('x')
    plt.ylabel('delta_th')

    # Heat Transfer Coefficient for Laminar Flow (alpha_lam) vs x
    plt.subplot(2, 2, 4)
    plt.plot(df['x'], df['alpha_lam'])
    plt.title('Heat Transfer Coefficient for Laminar Flow (alpha_lam) vs x')
    plt.xlabel('x')
    plt.ylabel('alpha_lam')

    plt.tight_layout()
    plt.show()

# Constants
nue = 0.000013  # unit not specified
lambda_ = 0.025  # W/(m K)
Pr = 0.7         # unit not specified
w_m_per_s = 10.0  # 10 m/s (convert from km/h if needed)

# Adjusted range of x values
x_values = np.arange(0, 4.1, 0.1)  # From 0 to 4 in steps of 0.1

# Generating the plots with the updated x values
plot_values(w_m_per_s, nue, lambda_, Pr, x_values)
