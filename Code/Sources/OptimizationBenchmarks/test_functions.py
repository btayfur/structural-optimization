"""
Test script for the optimization benchmark functions.

This script demonstrates how to use the benchmark functions 
and visualizes some common functions in 2D.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Import functions from their respective categories
from many_local_minima import ackley, rastrigin, schwefel
from bowl_shaped import sphere, sum_squares
from plate_shaped import booth, zakharov
from valley_shaped import rosenbrock
from steep_functions import easom
from other_functions import beale, branin


def evaluate_at_point(functions, x):
    """
    Evaluate multiple functions at a given point
    
    Parameters:
        functions (dict): Dictionary of function names and function objects
        x (array): Input vector
    """
    print(f"\nEvaluating functions at point: {x}")
    print("-" * 50)
    
    for name, func in functions.items():
        try:
            value = func(x)
            print(f"{name:20}: {value:.6f}")
        except Exception as e:
            print(f"{name:20}: Error - {e}")
    
    print("-" * 50)


def plot_2d_function(func, name, bounds=(-5, 5), points=100):
    """
    Create a 3D surface plot of a 2D function
    
    Parameters:
        func (callable): The function to plot
        name (str): Name of the function for the title
        bounds (tuple): (lower_bound, upper_bound) for both x and y
        points (int): Number of points in each dimension
    """
    x = np.linspace(bounds[0], bounds[1], points)
    y = np.linspace(bounds[0], bounds[1], points)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(points):
        for j in range(points):
            Z[i, j] = func([X[i, j], Y[i, j]])
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.8)
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.set_title(f'{name} Function')
    
    # Add contour plot at the bottom
    offset = np.min(Z) - 0.5 * (np.max(Z) - np.min(Z))
    contour = ax.contour(X, Y, Z, 20, cmap=cm.coolwarm, linestyles="solid", offset=offset)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run tests and examples"""
    # Define a set of test points
    test_points = [
        np.array([0.0, 0.0]),  # Global minimum for many functions
        np.array([1.0, 1.0]),  # Global minimum for Rosenbrock
        np.array([2.0, -3.0]),  # Random point
        np.array([3.14159, 3.14159])  # Pi values
    ]
    
    # Define functions to test
    test_functions = {
        "Ackley": ackley,
        "Sphere": sphere,
        "Rastrigin": rastrigin,
        "Rosenbrock": rosenbrock,
        "Booth": booth,
        "Beale": beale,
        "Schwefel": schwefel,
        "Easom": easom
    }
    
    # Evaluate functions at test points
    for point in test_points:
        evaluate_at_point(test_functions, point)
    
    # Visualize some functions
    print("\nCreating 3D plots for selected functions...")
    
    # Functions with different characteristics and plot bounds
    plot_functions = [
        (ackley, "Ackley", (-5, 5)),
        (rastrigin, "Rastrigin", (-5.12, 5.12)),
        (rosenbrock, "Rosenbrock", (-2, 2)),
        (sphere, "Sphere", (-3, 3)),
        (easom, "Easom", (-3, 6)),
        (branin, "Branin", (-5, 10))
    ]
    
    for func, name, bounds in plot_functions:
        plot_2d_function(func, name, bounds)


if __name__ == "__main__":
    main() 