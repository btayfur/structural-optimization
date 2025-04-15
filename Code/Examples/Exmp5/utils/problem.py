import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def binh_and_korn(x):
    """
    Binh and Korn function for multi-objective optimization.
    Input: x - numpy array with 2 elements [x1, x2]
    Output: [f1, f2] - Two objective function values
    
    Constraints:
    (x1 - 5)^2 + x2^2 <= 25
    (x1 - 8)^2 + (x2 + 3)^2 >= 7.7
    
    Both x1 and x2 are bounded within [0, 5] and [0, 3] respectively.
    """
    x1, x2 = x[0], x[1]
    
    # Objective functions
    f1 = 4 * x1**2 + 4 * x2**2
    f2 = (x1 - 5)**2 + (x2 - 5)**2
    
    # Check constraints
    c1 = (x1 - 5)**2 + x2**2 - 25
    c2 = 7.7 - (x1 - 8)**2 - (x2 + 3)**2
    
    # Add penalty if constraints are violated
    if c1 > 0 or c2 > 0:
        f1 += 1000 * max(0, c1) + 1000 * max(0, c2)
        f2 += 1000 * max(0, c1) + 1000 * max(0, c2)
    
    return np.array([f1, f2])

def plot_pareto_front(solutions, objectives, algorithm_name, save_path=None):
    """
    Plot a 3D visualization of solutions and their corresponding objective values.
    
    Parameters:
    - solutions: Array of solution points (x-space)
    - objectives: Array of objective values (f-space)
    - algorithm_name: String, name of the algorithm for the title
    - save_path: String, path to save the figure (if None, just display)
    """
    fig = plt.figure(figsize=(12, 10))
    
    # First subplot: 2D Pareto front
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(objectives[:, 0], objectives[:, 1], c='b', marker='o', alpha=0.6)
    ax1.set_title(f'Pareto Front - {algorithm_name}')
    ax1.set_xlabel('f1(x) - First Objective')
    ax1.set_ylabel('f2(x) - Second Objective')
    ax1.grid(True)
    
    # Second subplot: 3D visualization
    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    
    # Plot the solutions in decision space along with their objective values (color-coded)
    p = ax2.scatter(solutions[:, 0], solutions[:, 1], objectives[:, 0], 
                   c=objectives[:, 1], cmap=cm.viridis, marker='o', alpha=0.6)
    
    ax2.set_title(f'Solution Space to Objective Space - {algorithm_name}')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('f1(x)')
    
    fig.colorbar(p, ax=ax2, label='f2(x) value')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def get_true_pareto_front(num_points=100):
    """
    Generate an approximation of the true Pareto front for the Binh and Korn problem
    using a dense, even sampling approach.
    
    Parameters:
    - num_points: Number of points to generate for the Pareto front
    
    Returns:
    - Array of points on the Pareto front
    """
    # Generate a fine grid of points in the decision space
    x1_values = np.linspace(0, 5, 100)
    x2_values = np.linspace(0, 3, 100)
    
    # Generate all combinations of x1 and x2
    all_points = []
    all_objectives = []
    
    for x1 in x1_values:
        for x2 in x2_values:
            # Check if point satisfies constraints
            c1 = (x1 - 5)**2 + x2**2 - 25
            c2 = 7.7 - (x1 - 8)**2 - (x2 + 3)**2
            
            if c1 <= 0 and c2 <= 0:  # If constraints are satisfied
                point = np.array([x1, x2])
                obj = binh_and_korn(point)
                all_points.append(point)
                all_objectives.append(obj)
    
    # Convert to numpy arrays
    all_objectives = np.array(all_objectives)
    
    # Find the non-dominated points
    pareto_indices = []
    for i, obj_i in enumerate(all_objectives):
        is_dominated = False
        for obj_j in all_objectives:
            # Check if obj_j dominates obj_i
            if np.all(obj_j <= obj_i) and np.any(obj_j < obj_i):
                is_dominated = True
                break
        if not is_dominated:
            pareto_indices.append(i)
    
    pareto_front = all_objectives[pareto_indices]
    
    # Sort by first objective for better visualization
    sorted_indices = np.argsort(pareto_front[:, 0])
    pareto_front = pareto_front[sorted_indices]
    
    # If there are too many points, reduce the number
    if len(pareto_front) > num_points:
        step = len(pareto_front) // num_points
        pareto_front = pareto_front[::step]
    
    return pareto_front 