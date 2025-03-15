# **********************************************************************
# Constrained Optimization BY Linear Approximations (COBYLA) ----
#
# Purpose ----
# Perform function optimization using the Constrained Optimization BY Linear
# Approximations (COBYLA) algorithm.
# **********************************************************************

# Imports ----
import nlopt
import numpy as np


# Objective Function ----
def objective_function(x, grad):
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
def g1(x, grad): return x[17] - x[0]


def g2(x, grad): return ((x[1] * np.sin((2 * np.pi) / x[2] * (x[3] - x[4])) + x[5]) * x[6] * x[7]) - x[0]


def g3(x, grad): return (x[8] * x[6] * x[9]) - x[0]


def g4(x, grad): return np.abs((x[12] * np.sin((2 * np.pi) / x[13] * (x[14] - x[15])) + x[16]) + x[10]) - 1000


def g5(x, grad): return x[0] - (x[18] + x[19])


def g6(x, grad): return x[11]


def g7(x, grad): return x[11] + x[10] - 1000  # Inactive (removed)


def g8(x, grad): return - x[17]


def g9(x, grad): return x[18] - x[19]  # Inactive (removed)


# Bounds: Lower-Bound (lb), Upper-Bound (ub), and Initial Point (IP) ----
lower_bounds = np.array([34.5799, 9999.9999, 5.9999, 5.9999, 4.4999, 26305.1399, 0.0999,
               0.00034, 4.9999, 1.4899, 719.9999, 719.9999, 179.9999, 5.9999,
               5.9999, 4.4999, 719.9999, 36.9999, 0.3699, 0.3699])
upper_bounds = np.array([345.68, 10000.1, 6.1, 6.1, 4.6, 26305.24, 20.1, 0.10044, 100.1,
               1.59, 3600.1, 3600.1, 675.1, 6.1, 6.1, 4.6, 2700.1, 7400.1,
               148.1, 444.52])
init_point = np.array([190.08,  10000,  6,  6,  4.5,  26305.14,  10.05,  0.00044,
               52.5,  1.49,  2160,  2160,  427.5,  6,  6,  4.5,  1710,  3718.5,
               74.185, 222.395])

# Optimizer Object ----
# Create an optimizer object with 20 dimensions
opt = nlopt.opt(nlopt.LN_COBYLA, 20)

# Set the objective function
opt.set_min_objective(objective_function)

# Add the inequality constraints
# with a tolerance for how closely they must be met
opt.add_inequality_constraint(g1, 1e-3)
opt.add_inequality_constraint(g2, 1e-3)
opt.add_inequality_constraint(g3, 1e-3)
opt.add_inequality_constraint(g4, 1e-3)
opt.add_inequality_constraint(g5, 1e-3)
opt.add_inequality_constraint(g6, 1e-3)
# opt.add_inequality_constraint(g7, 1e-3)
opt.add_inequality_constraint(g8, 1e-3)
# opt.add_inequality_constraint(g9, 1e-3)

# Add the equality constraints
# with a tolerance for how closely they must be met
# opt.add_equality_constraint(h1, 1e0)

# Set the lower bounds and upper bounds
opt.set_lower_bounds(lower_bounds)
opt.set_upper_bounds(upper_bounds)

# Set stopping criteria
opt.set_xtol_rel(1e-3)
opt.set_maxeval(100000)

# Perform the Optimization ----
# Initialization point
x_opt = opt.optimize(init_point)
min_f = opt.last_optimum_value()

# Print the Objective Function Value at the Optimal Solution ----
# print(f"Optimal solution: {x_opt}, Objective function value at optimal solution: {min_f}")

x_opt_formatted = ", ".join([f"{x:.8f}" for x in x_opt])
print(f"Optimal solution: [{x_opt_formatted}], Objective function value at optimal solution: {min_f:.8f}")
