# **********************************************************************
# Particle Swarm Optimization (PSO) ----
#
# Purpose ----
# Perform function optimization using the Particle Swarm Optimization
# (PSO) algorithm.
# **********************************************************************

# Imports ----
import numpy as np
from pyswarm import pso


# Objective Function ----
def objective_function(x):
    """
    The variables and parameters have been coded as follows:
    x[0] = r
    x[1] = amp_w
    x[2] = t_w
    x[3] = tp_w
    x[4] = phase_w
    x[5] = vert_w
    x[6] = acre
    x[7] = c_w
    x[8] = qe
    x[9] = ce
    x[10] = qd
    x[11] = qs
    x[12] = amp_s
    x[13] = t_s
    x[14] = tp_s
    x[15] = phase_s
    x[16] = vert_s
    x[17] = tal
    x[18] = exp
    x[19] = mal
    """
    return (
        0.2350747 * x[0] ** (-1.0)
        + 0.4804318 * (
            (((x[1] * np.sin((2 * np.pi) / x[2] * (x[3] - x[4])) + x[5]) * x[6] * x[7]) ** 0.4) * 
            (x[8] * x[6] * x[9]) ** 0.6 
        )
        + 0.2811869 * x[10]
        - 0.9963252 * x[11]
        - 0.1230044 * ((x[12] * np.sin((2 * np.pi) / x[13] * (x[14] - x[15])) + x[16]) - x[10])
        + 0.2777817 * x[17] ** (-1.0)
        + 1.1544897 * x[16] ** (-1.0)
        + 0.1500959 * x[18] ** (-1.0)
        + 0.1491099 * x[19] ** (-1.0)
        + 0.0004785
    )


# Constraint Functions ----
def g1(x): return x[17] - x[0]


def g2(x): return ((x[1] * np.sin((2 * np.pi) / x[2] * (x[3] - x[4])) + x[5]) * x[6] * x[7]) - x[0]


def g3(x): return (x[8] * x[6] * x[9]) - x[0]


def g4(x): return np.abs((x[12] * np.sin((2 * np.pi) / x[13] * (x[14] - x[15])) + x[16]) + x[10]) - 1000


def g5(x): return x[0] - (x[18] + x[19])


def g6(x): return x[11]


def g7(x): return x[11] + x[10] - 1000  # Inactive (removed)


def g8(x): return - x[17]


def g9(x): return x[18] - x[19]  # Inactive (removed)


# Constraints passed as a list of functions
constraints = [g1, g2, g3, g4, g5, g6]

# Bounds: Lower-Bound (lb), Upper-Bound (ub), and Initial Point (IP) ----
lb = np.array([34.5799, 9999.9999, 5.9999, 5.9999, 4.4999, 26305.1399, 0.0999,
               0.00034, 4.9999, 1.4899, 719.9999, 719.9999, 179.9999, 5.9999,
               5.9999, 4.4999, 719.9999, 36.9999, 0.3699, 0.3699])
ub = np.array([345.68, 10000.1, 6.1, 6.1, 4.6, 26305.24, 20.1, 0.10044, 100.1,
               1.59, 3600.1, 3600.1, 675.1, 6.1, 6.1, 4.6, 2700.1, 7400.1,
               148.1, 444.52])
ip = np.array([190.08,  10000,  6,  6,  4.5,  26305.14,  10.05,  0.00044,
               52.5,  1.49,  2160,  2160,  427.5,  6,  6,  4.5,  1710,  3718.5,
               74.185, 222.395])

# Perform the Optimization ----
xopt, fopt = pso(objective_function, lb, ub, ieqcons=constraints,
                 f_ieqcons=None, maxiter=100000, swarmsize=100, debug=True)

# Print the Objective Function Value at the Optimal Solution ----
# print("Optimal solution:", xopt)
# print("Objective function value at optimal solution:", fopt)

# print("Optimal solution:")
# for value in xopt:
#     print(f"{value:.8f}")
# print(f"Objective function value at optimal solution: {fopt:.8f}")
print(f"Optimal solution: {', '.join(f'{value:.8f}' for value in xopt)}"
      f", Objective function value at optimal solution: {fopt:.8f}")

# Optimal Solution ----
# Optimal solution leads to an objective function value of -3,515.885829
optimized_x = np.array([197.3555162, 10000.04094, 6.06058071, 6.06481227,
                        4.55822528, 26305.20034, 4.36023396, 0.00221809,
                        55.86751226, 1.54009833, 720.005943, 3600.099775,
                        524.0792289, 6.04250929, 6.03679917, 4.55245179,
                        2082.596513, 4423.625629, 52.0247252, 51.7242541])
