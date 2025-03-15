# **********************************************************************
# Particle Swarm Optimization (PSO) ----
#
# Purpose ----
# To support decision-making by understanding which variables have the most
# significant impact on the outcome and may require closer attention or more
# accurate estimation in practical applications.
# **********************************************************************

# Imports ----
import numpy as np

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

# Optimal Solution ----
# Optimal solution leads to an objective function value of -3,515.885829
optimized_x = np.array([197.3555162, 10000.04094, 6.06058071, 6.06481227,
                        4.55822528, 26305.20034, 4.36023396, 0.00221809,
                        55.86751226, 1.54009833, 720.005943, 3600.099775,
                        524.0792289, 6.04250929, 6.03679917, 4.55245179,
                        2082.596513, 4423.625629, 52.0247252, 51.7242541])

variables_of_interest_indices = [16, 17, 18, 19]
perturbation_percentage = 0.15  # 15% perturbation

# Sensitivity Analysis ----
def perform_sensitivity_analysis(optimized_x, variables_indices, perturbation_percentage):
    # Store the results
    results = {}
    
    for var_index in variables_indices:
        # Perturb the variable up and down by the specified percentage
        perturbation = optimized_x[var_index] * perturbation_percentage
        x_perturbed_up = optimized_x.copy()
        x_perturbed_down = optimized_x.copy()
        
        x_perturbed_up[var_index] += perturbation
        x_perturbed_down[var_index] -= perturbation
        
        # Calculate the objective function value for the perturbed values
        obj_value_up = objective_function(x_perturbed_up)
        obj_value_down = objective_function(x_perturbed_down)
        
        # Store the results
        results[f'x[{var_index}]'] = {
            'Perturbation': perturbation,
            'Objective Value (Perturbed Up)': obj_value_up,
            'Objective Value (Perturbed Down)': obj_value_down,
            'Change (Perturbed Up)': obj_value_up - objective_function(optimized_x),
            'Change (Perturbed Down)': obj_value_down - objective_function(optimized_x),
        }
    
    return results

# Performing the sensitivity analysis to get the results
# sensitivity_results_full = perform_sensitivity_analysis(
#     optimized_x, variables_of_interest_indices,
#     perturbation_percentage)

# print(sensitivity_results_full)

sensitivity_results_full = perform_sensitivity_analysis(
    optimized_x, variables_of_interest_indices, perturbation_percentage)

# Print the Results ----
# Printing the results
for var, details in sensitivity_results_full.items():
    print(f"\n{var}:")  # Print the variable name with a newline before it
    for key, value in details.items():
        print(f"  {key}: {value}")  # Print each detail indented for readability
