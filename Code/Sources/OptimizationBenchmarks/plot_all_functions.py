"""
Script to plot all 2D benchmark functions and save them to the graphs folder.

This script generates 3D surface plots for all benchmark functions that support 2D input 
and saves them to the graphs folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Import functions from their respective categories
# Many local minima functions
from many_local_minima import (
    ackley, bukin_n6, cross_in_tray, drop_wave, eggholder, griewank, holder_table, 
    levy_n13, rastrigin, schaffer_n2, schaffer_n4, schwefel, shubert
)

# Bowl-shaped functions
from bowl_shaped import (
    bohachevsky, sphere, sum_of_different_powers, sum_squares, trid
)

# Plate-shaped functions
from plate_shaped import (
    booth, matyas, mccormick, zakharov
)

# Valley-shaped functions
from valley_shaped import (
    three_hump_camel, six_hump_camel, dixon_price, rosenbrock
)

# Steep functions
from steep_functions import (
    easom, michalewicz
)

# Other functions
from other_functions import (
    beale, branin, goldstein_price
)


def plot_2d_function(func, name, bounds=(-5, 5), points=100, cmap=cm.coolwarm):
    """
    Create a 3D surface plot of a 2D function and save it to the graphs folder
    
    Parameters:
        func (callable): The function to plot
        name (str): Name of the function for the title
        bounds (tuple): (lower_bound, upper_bound) for both x and y
        points (int): Number of points in each dimension
        cmap: Matplotlib colormap
    """
    # Create graphs directory if it doesn't exist
    os.makedirs("graphs", exist_ok=True)
    
    x = np.linspace(bounds[0], bounds[1], points)
    y = np.linspace(bounds[0], bounds[1], points)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(points):
        for j in range(points):
            try:
                Z[i, j] = func([X[i, j], Y[i, j]])
            except Exception as e:
                print(f"Error computing {name} at [{X[i, j]}, {Y[i, j]}]: {e}")
                return False
    
    # Clip extremely large values for better visualization
    vmin = np.percentile(Z, 2)
    vmax = np.percentile(Z, 98)
    Z_clipped = np.clip(Z, vmin, vmax)
    
    # Create 3D plot
    plt.figure(figsize=(10, 8))
    
    # Create two subplots: one for 3D, one for contour
    ax1 = plt.subplot(121, projection='3d')
    ax2 = plt.subplot(122)
    
    # Plot the surface
    surf = ax1.plot_surface(X, Y, Z_clipped, cmap=cmap, linewidth=0, antialiased=True, alpha=0.8)
    
    # Set labels and title for 3D plot
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X, Y)')
    
    # Create contour plot
    contour = ax2.contourf(X, Y, Z_clipped, 30, cmap=cmap)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(contour, ax=ax2, shrink=0.8)
    
    # Set overall title
    plt.suptitle(f'{name} Function', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to make room for the title
    
    # Save the figure
    filename = f"graphs/{name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()
    
    return True


def main():
    """Main function to plot all 2D functions"""
    
    # Dictionary of all 2D functions with appropriate bounds
    functions = {
        # Many local minima
        "Ackley": (ackley, (-5, 5)),
        "Bukin N6": (bukin_n6, (-15, 5), (-3, 3)),
        "Cross-in-Tray": (cross_in_tray, (-10, 10)),
        "Drop-Wave": (drop_wave, (-5.12, 5.12)),
        "Eggholder": (eggholder, (-512, 512)),
        "Griewank": (griewank, (-50, 50)),
        "Holder Table": (holder_table, (-10, 10)),
        "Levy N13": (levy_n13, (-10, 10)),
        "Rastrigin": (rastrigin, (-5.12, 5.12)),
        "Schaffer N2": (schaffer_n2, (-100, 100)),
        "Schaffer N4": (schaffer_n4, (-100, 100)),
        "Schwefel": (schwefel, (-500, 500)),
        "Shubert": (shubert, (-10, 10)),
        
        # Bowl-shaped
        "Bohachevsky": (lambda x: bohachevsky(x, 1), (-100, 100)),
        "Sphere": (sphere, (-5.12, 5.12)),
        "Sum of Different Powers": (sum_of_different_powers, (-1, 1)),
        "Sum Squares": (sum_squares, (-10, 10)),
        "Trid": (trid, (-20, 20)),
        
        # Plate-shaped
        "Booth": (booth, (-10, 10)),
        "Matyas": (matyas, (-10, 10)),
        "McCormick": (mccormick, (-3, 4), (-3, 4)),
        "Zakharov": (zakharov, (-5, 10)),
        
        # Valley-shaped
        "Three-Hump Camel": (three_hump_camel, (-5, 5)),
        "Six-Hump Camel": (six_hump_camel, (-3, 3), (-2, 2)),
        "Dixon-Price": (dixon_price, (-10, 10)),
        "Rosenbrock": (rosenbrock, (-5, 10)),
        
        # Steep functions
        "Easom": (easom, (-10, 10)),
        "Michalewicz": (lambda x: michalewicz(x, 10), (0, np.pi)),
        
        # Other functions
        "Beale": (beale, (-4.5, 4.5)),
        "Branin": (branin, (-5, 15), (0, 15)),
        "Goldstein-Price": (goldstein_price, (-2, 2))
    }
    
    success_count = 0
    total_count = len(functions)
    
    print(f"Plotting {total_count} 2D benchmark functions...")
    
    # Plot each function with appropriate bounds
    for name, func_info in functions.items():
        print(f"Processing {name}...")
        
        if len(func_info) == 2:
            func, bounds = func_info
            x_bounds = bounds
            y_bounds = bounds
        else:
            func, x_bounds, y_bounds = func_info
        
        # Determine common bounds for the plot
        bounds = (min(x_bounds[0], y_bounds[0]), max(x_bounds[1], y_bounds[1]))
        
        # Different colormaps for different function categories
        if "Rosenbrock" in name or "Camel" in name:
            cmap = cm.viridis
        elif "Ackley" in name or "Rastrigin" in name or "Schwefel" in name:
            cmap = cm.jet
        elif "Easom" in name or "Michalewicz" in name:
            cmap = cm.plasma
        else:
            cmap = cm.coolwarm
            
        if plot_2d_function(func, name, bounds, points=100, cmap=cmap):
            success_count += 1
    
    print(f"Successfully plotted {success_count} out of {total_count} functions.")
    print(f"Graphs saved to the 'graphs' folder.")


if __name__ == "__main__":
    main() 