# **********************************************************************
# Visualization Code ----
#
# Purpose ----
# To create a 3D visualization of the objective function.
# **********************************************************************

# Imports ----
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Objective Function ----
def objective_function(exp, mal):
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

    # Initialize x with ones for other variables, and set the last two to exp and mal
    x = np.ones(20)
    x[18] = exp  # exp
    x[19] = mal  # mal
    return (
        0.2350747 * x[0] ** (-1.0)
        + 0.4804318 * (
            (((x[1] * np.sin((2 * np.pi) / x[2] * (x[3] - x[4])) + x[5]) * x[6] * x[7]) ** 0.4) * 
            (x[8] * x[6] * x[9]) ** 0.6 
        )
        + 0.2811869 * x[10]
        - 0.9963252 * x[11]
        - 0.1230044 * ((x[12] * np.sin((2 * np.pi) / x[14] * (x[14] - x[15])) + x[16]) * x[10])
        + 0.2777817 * x[17] ** (-1.0)
        + 1.1544897 * x[16] ** (-1.0)
        + 0.1500959 * x[18] ** (-1.0)
        + 0.1491099 * x[19] ** (-1.0)
        + 0.0004785
    )

# Create the grid of values
exp_values = np.linspace(0.37, 2, 148)  # Avoid division by zero by starting from USD 0.37
mal_values = np.linspace(0.37, 2, 148)  # Avoid division by zero by starting from USD 0.37
exp_grid, mal_grid = np.meshgrid(exp_values, mal_values)

# Vectorize the objective function for element-wise application
vectorized_objective_function = np.vectorize(objective_function)

# Apply the vectorized function over the grid
z_values = vectorized_objective_function(exp_grid, mal_grid)

# Plotting the 3D surface plot
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(exp_grid, mal_grid, z_values, cmap='jet')

# Labels and title
ax.set_xlabel('Export Fee (exp)')
ax.set_ylabel('Market Access Licence Fee (mal)')
ax.set_zlabel('Objective Function')
ax.set_title('Surface Plot of the Unconstrained Objective Function\'s \nMarket Access License Fee (mal) \nand the Export Fee (exp) \nWhile Holding all Other Variables at a Constant of 1')

# Color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

# Show the plot
plt.show()
