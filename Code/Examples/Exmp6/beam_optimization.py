import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse
import os

# Constants
L = 3.0  # Total length of beam in meters
num_segments = 5  # Number of segments
segment_length = L / num_segments  # Length of each segment
E = 210e9  # Young's modulus for S270 steel (Pa)
rho = 7850  # Density of steel (kg/m^3)
sigma_yield = 270e6  # Yield strength of S270 steel (Pa)
max_displacement = 0.02  # Maximum displacement at the end (m)

# Force applied at the end of the beam (N)
F = 500000  # Increased force value to create more realistic displacement

def calculate_area_moment_of_inertia(r_outer, r_inner):
    """Calculate the second moment of area for a hollow circular cross section."""
    return np.pi * (r_outer**4 - r_inner**4) / 4

def calculate_area(r_outer, r_inner):
    """Calculate the cross-sectional area of a hollow circular cross section."""
    return np.pi * (r_outer**2 - r_inner**2)

def assemble_stiffness_matrix(params):
    """Assemble the global stiffness matrix for the beam."""
    # Extract radii from parameters (convert from cm to m)
    r_outer = params[0:num_segments] / 100
    r_inner = params[num_segments:2*num_segments] / 100
    
    # Initialize the global stiffness matrix
    # For a beam with n segments, we have n+1 nodes and 2 DOFs per node (displacement and rotation)
    K = np.zeros((2*(num_segments+1), 2*(num_segments+1)))
    
    # Assemble the global stiffness matrix
    for i in range(num_segments):
        # Calculate the element stiffness matrix
        I = calculate_area_moment_of_inertia(r_outer[i], r_inner[i])
        l = segment_length
        
        # Element stiffness matrix for beam element (4x4)
        k_e = np.array([
            [12*E*I/l**3, 6*E*I/l**2, -12*E*I/l**3, 6*E*I/l**2],
            [6*E*I/l**2, 4*E*I/l, -6*E*I/l**2, 2*E*I/l],
            [-12*E*I/l**3, -6*E*I/l**2, 12*E*I/l**3, -6*E*I/l**2],
            [6*E*I/l**2, 2*E*I/l, -6*E*I/l**2, 4*E*I/l]
        ])
        
        # Add the element stiffness to the global stiffness matrix
        # Local DOFs: [u_i, θ_i, u_i+1, θ_i+1]
        # Global DOFs: [u_1, θ_1, u_2, θ_2, ..., u_n+1, θ_n+1]
        dofs = [2*i, 2*i+1, 2*i+2, 2*i+3]
        for j in range(4):
            for k in range(4):
                K[dofs[j], dofs[k]] += k_e[j, k]
    
    return K

def calculate_displacement(params):
    """Calculate the displacement at the end of the beam."""
    K = assemble_stiffness_matrix(params)
    
    # Apply boundary conditions (fixed at left end)
    # Remove rows and columns corresponding to the fixed DOFs
    K_reduced = K[2:, 2:]
    
    # Force vector (force applied at the tip)
    F_vector = np.zeros(2*(num_segments+1)-2)
    F_vector[0] = F  # Apply force at the first free displacement DOF
    
    # Solve for displacements
    try:
        displacements = np.linalg.solve(K_reduced, F_vector)
        # Return the displacement at the beam end (last node, vertical DOF)
        return displacements[-2]  # -2 because every node has 2 DOFs (vertical, rotation)
    except np.linalg.LinAlgError:
        # Return a large value if the matrix is singular
        return 1e10

def calculate_stresses(params, displacements):
    """Calculate the maximum stress in each segment."""
    # Extract radii from parameters (convert from cm to m)
    r_outer = params[0:num_segments] / 100
    r_inner = params[num_segments:2*num_segments] / 100
    
    # Include the fixed DOFs in the displacement vector
    full_displacements = np.zeros(2*(num_segments+1))
    full_displacements[2:] = displacements
    
    stresses = np.zeros(num_segments)
    
    for i in range(num_segments):
        # Get the displacements and rotations at the nodes of this element
        dofs = [2*i, 2*i+1, 2*i+2, 2*i+3]
        element_displacements = full_displacements[dofs]
        
        # Calculate the strain at the outer fiber
        l = segment_length
        c = r_outer[i]  # Distance from neutral axis to outer fiber
        I = calculate_area_moment_of_inertia(r_outer[i], r_inner[i])
        
        # Calculate second derivative of displacement (curvature)
        # This is a simplification - in practice you might need a more accurate method
        curvature = 6*E*I/l**2 * (element_displacements[0] - element_displacements[2]) + \
                    2*E*I/l * (2*element_displacements[1] + element_displacements[3])
        
        # Calculate stress from curvature
        stress = E * c * curvature / I
        stresses[i] = abs(stress)
    
    return stresses

def objective_function(params):
    """Calculate the weight of the beam (to be minimized)."""
    # Extract radii from parameters (convert from cm to m)
    r_outer = params[0:num_segments] / 100
    r_inner = params[num_segments:2*num_segments] / 100
    
    # Calculate volume and weight
    volume = 0
    for i in range(num_segments):
        area = calculate_area(r_outer[i], r_inner[i])
        volume += area * segment_length
    
    weight = volume * rho
    return weight

def constraint_displacement(params):
    """Constraint: displacement at the end must be less than the maximum allowed."""
    displacement = calculate_displacement(params)
    return max_displacement - abs(displacement)

def constraint_radii(params):
    """Constraint: outer radius must be greater than inner radius for each segment."""
    r_outer = params[0:num_segments]
    r_inner = params[num_segments:2*num_segments]
    
    constraints = []
    for i in range(num_segments):
        constraints.append(r_outer[i] - r_inner[i])
    
    return np.array(constraints)

def constraint_welding(params):
    """Constraint: inner radius of preceding segment must be smaller than outer radius of next segment."""
    r_outer = params[0:num_segments]
    r_inner = params[num_segments:2*num_segments]
    
    constraints = []
    for i in range(num_segments-1):
        constraints.append(r_outer[i+1] - r_inner[i])
    
    return np.array(constraints)

def constraint_yield_stress(params):
    """Constraint: stress must be less than yield stress."""
    # Extract radii from parameters (convert from cm to m)
    r_outer = params[0:num_segments] / 100
    r_inner = params[num_segments:2*num_segments] / 100
    
    # Calculate displacements
    K = assemble_stiffness_matrix(params)
    K_reduced = K[2:, 2:]
    F_vector = np.zeros(2*(num_segments+1)-2)
    F_vector[0] = F  # Apply force at the first free displacement DOF
    
    try:
        displacements = np.linalg.solve(K_reduced, F_vector)
    except np.linalg.LinAlgError:
        return -1e10  # Return a large negative value if the matrix is singular
    
    # Calculate maximum stress in each segment
    max_stress = 0
    for i in range(num_segments):
        # Distance from neutral axis to the outer fiber
        c = r_outer[i]
        
        # Calculate moment at this segment
        # For a cantilever beam with point load at the end, moment decreases linearly from the fixed end
        x = (i + 0.5) * segment_length  # Position at the middle of the segment
        moment = F * (L - x)
        
        # Calculate stress using beam bending formula
        I = calculate_area_moment_of_inertia(r_outer[i], r_inner[i])
        stress = moment * c / I
        
        max_stress = max(max_stress, abs(stress))
    
    return sigma_yield - max_stress

def simulated_annealing(initial_params, n_iterations=5000, initial_temp=100.0, cooling_rate=0.99):
    """Optimize the beam using Simulated Annealing algorithm."""
    current_params = initial_params.copy()
    best_params = current_params.copy()
    current_weight = objective_function(current_params)
    best_weight = current_weight
    
    # Check if initial parameters satisfy constraints
    displacement_ok = constraint_displacement(current_params) >= 0
    radii_ok = all(constraint_radii(current_params) >= 0)
    welding_ok = all(constraint_welding(current_params) >= 0)
    stress_ok = constraint_yield_stress(current_params) >= 0
    
    initial_feasible = displacement_ok and radii_ok and welding_ok and stress_ok
    
    if not initial_feasible:
        print("Initial parameters violate constraints, trying to find a feasible starting point...")
        
        # Try to find a feasible starting point
        for _ in range(1000):
            # Increase all dimensions for a safer start
            trial_params = initial_params * (1 + np.random.uniform(0, 1, len(initial_params)))
            
            displacement_ok = constraint_displacement(trial_params) >= 0
            radii_ok = all(constraint_radii(trial_params) >= 0)
            welding_ok = all(constraint_welding(trial_params) >= 0)
            stress_ok = constraint_yield_stress(trial_params) >= 0
            
            if displacement_ok and radii_ok and welding_ok and stress_ok:
                current_params = trial_params.copy()
                best_params = trial_params.copy()
                current_weight = objective_function(trial_params)
                best_weight = current_weight
                print(f"Found feasible starting point with weight: {best_weight:.2f} kg")
                break
        else:
            print("Warning: Could not find a feasible starting point. Optimization may fail.")
    
    temp = initial_temp
    
    # Keep track of best solution history
    history = []
    
    # Track consecutive rejections to adjust step size
    consecutive_rejections = 0
    step_size = 1.0
    
    for i in range(n_iterations):
        # Adjust step size based on acceptance rate
        if consecutive_rejections > 50:
            step_size *= 0.9
            consecutive_rejections = 0
        
        # Generate a random neighbor
        # Use smaller perturbations for more refined search
        perturbation = np.random.uniform(-step_size, step_size, len(current_params))
        neighbor_params = current_params + perturbation
        
        # Ensure all radii are positive
        neighbor_params = np.maximum(neighbor_params, 1.0)
        
        # Check constraints
        displacement_ok = constraint_displacement(neighbor_params) >= 0
        radii_ok = all(constraint_radii(neighbor_params) >= 0)
        welding_ok = all(constraint_welding(neighbor_params) >= 0)
        stress_ok = constraint_yield_stress(neighbor_params) >= 0
        
        if displacement_ok and radii_ok and welding_ok and stress_ok:
            # Calculate new weight
            neighbor_weight = objective_function(neighbor_params)
            
            # Decide whether to accept the new solution
            if neighbor_weight < current_weight:
                # Accept better solution
                current_params = neighbor_params.copy()
                current_weight = neighbor_weight
                consecutive_rejections = 0
                
                # Update best solution if needed
                if current_weight < best_weight:
                    best_params = current_params.copy()
                    best_weight = current_weight
            else:
                # Accept worse solution with some probability
                delta = neighbor_weight - current_weight
                probability = np.exp(-delta / temp)
                
                if np.random.random() < probability:
                    current_params = neighbor_params.copy()
                    current_weight = neighbor_weight
                    consecutive_rejections = 0
                else:
                    consecutive_rejections += 1
        else:
            consecutive_rejections += 1
        
        # Cool down
        temp = initial_temp * (cooling_rate ** i)
        
        # Save history
        history.append(best_weight)
        
        if i % 500 == 0:
            print(f"Iteration {i}, Best Weight: {best_weight:.2f} kg, Temperature: {temp:.6f}")
    
    return best_params, best_weight, history

def plot_deformed_beam(params, filename="deformed_beam.png"):
    """Plot the deformed shape of the beam."""
    # Extract radii from parameters (convert from cm to m)
    r_outer = params[0:num_segments] / 100
    r_inner = params[num_segments:2*num_segments] / 100
    
    # Calculate displacements
    K = assemble_stiffness_matrix(params)
    K_reduced = K[2:, 2:]
    F_vector = np.zeros(2*(num_segments+1)-2)
    F_vector[0] = F  # Apply force at the first free displacement DOF
    
    displacements = np.linalg.solve(K_reduced, F_vector)
    
    # Include the fixed DOFs in the displacement vector
    full_displacements = np.zeros(2*(num_segments+1))
    full_displacements[2:] = displacements
    
    # Extract vertical displacements at each node
    # Negate the displacements to show downward deflection (positive force causes negative deflection)
    vertical_disp = -full_displacements[0::2]  # Negate to show downward deflection
    
    # Create x coordinates for nodes
    x_coords = np.linspace(0, L, num_segments+1)
    
    # Create a smooth curve for the deformed shape
    x_smooth = np.linspace(0, L, 100)
    
    # Use cubic interpolation for a smooth curve
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x_coords, vertical_disp)
    y_smooth = cs(x_smooth)
    
    # Plot simplified deformed shape
    plt.figure(figsize=(12, 6))
    
    # Plot original beam axis
    plt.plot(x_coords, np.zeros_like(x_coords), 'k-', linewidth=2, label='Original')
    
    # Plot smooth deformed shape
    plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Deformed')
    
    # Plot markers at the nodes
    plt.plot(x_coords, vertical_disp, 'ro', markersize=6)
    
    # Add annotations for displacement values at each node
    for i, (x, y) in enumerate(zip(x_coords, vertical_disp)):
        plt.annotate(f"{abs(y)*100:.2f} cm", (x, y), textcoords="offset points", 
                     xytext=(0,-15), ha='center', fontsize=9, 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add force arrow (now pointing downward)
    plt.arrow(L, 0, 0, vertical_disp[-1]/2, head_width=0.05, head_length=abs(vertical_disp[-1]/4), 
              fc='blue', ec='blue', linewidth=1.5)
    plt.text(L+0.05, vertical_disp[-1]/2, f"F = {F/1000:.0f} kN", 
             color='blue', fontsize=10, ha='left', va='center')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Beam Displacement')
    plt.xlabel('Length (m)')
    plt.ylabel('Displacement (m)')
    plt.legend(loc='upper right')
    
    # Set proper axis limits to show all displacement points
    min_disp = min(vertical_disp)
    y_min = min_disp * 1.2
    y_max = max(0, max(vertical_disp)) * 1.2
    
    # Add a small buffer to make sure all points are visible
    y_range = abs(y_min - y_max)
    if y_range < 0.001:  # Very small displacements
        y_buffer = 0.001
    else:
        y_buffer = y_range * 0.1
    
    plt.xlim(0, L+0.2)
    plt.ylim(y_min - y_buffer, y_max + y_buffer)
    
    # Get the script's directory and save the plot there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return vertical_disp

def plot_optimized_beam(params, filename="optimized_beam.png"):
    """Plot the optimized beam with improved visualization."""
    # Extract radii from parameters (convert from cm to m)
    r_outer = params[0:num_segments] / 100
    r_inner = params[num_segments:2*num_segments] / 100
    
    plt.figure(figsize=(14, 8))
    
    # Create a colormap for the segments
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0.2, 0.4, 0.6), (0.4, 0.6, 0.8)]  # Blue gradient
    cmap = LinearSegmentedColormap.from_list("beam_cmap", colors, N=num_segments)
    
    # Plot fixed end support
    plt.plot([-0.1, 0], [0, 0], 'k-', linewidth=4)
    plt.plot([-0.1, -0.1], [-0.25, 0.25], 'k-', linewidth=4)
    plt.plot([-0.1, 0], [-0.25, 0], 'k-', linewidth=4)
    plt.plot([-0.1, 0], [0.25, 0], 'k-', linewidth=4)
    
    # Plot beam axis
    plt.plot([0, L], [0, 0], 'k-', linewidth=2, label='Beam Axis')
    
    # Plot cross sections at regular intervals
    intervals = 5  # Number of cross sections to draw per segment
    for i in range(num_segments):
        # Get segment color
        segment_color = cmap(i/num_segments)
        
        for j in range(intervals + 1):
            # Position along the segment
            x = i * segment_length + j * segment_length / intervals
            
            # Draw cross section
            theta = np.linspace(0, 2*np.pi, 100)
            x_outer = x + r_outer[i] * np.cos(theta)
            y_outer = r_outer[i] * np.sin(theta)
            x_inner = x + r_inner[i] * np.cos(theta)
            y_inner = r_inner[i] * np.sin(theta)
            
            # Draw filled cross section with transparency
            if j == 0 or j == intervals:  # Draw full cross section at segment boundaries
                plt.fill(x_outer, y_outer, color=segment_color, alpha=0.4)
                plt.fill(x_inner, y_inner, color='white')
                plt.plot(x_outer, y_outer, color=segment_color, linewidth=1.5)
                plt.plot(x_inner, y_inner, color=segment_color, linewidth=1.5)
            elif j == intervals // 2:  # Draw outline in the middle
                plt.plot(x_outer, y_outer, color=segment_color, linewidth=1, linestyle='-')
                plt.plot(x_inner, y_inner, color=segment_color, linewidth=1, linestyle='-')
    
    # Add segment labels
    for i in range(num_segments):
        x = (i + 0.5) * segment_length
        plt.text(x, -0.05, f"Segment {i+1}", ha='center', va='top', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Add dimension lines and text for radii
        x_pos = i * segment_length + segment_length / 2
        y_inner = 0
        y_outer = 0
        
        # Draw dimension lines
        plt.plot([x_pos, x_pos], [y_inner, -r_inner[i]], 'k--', linewidth=0.5)
        plt.plot([x_pos, x_pos], [y_outer, r_outer[i]], 'k--', linewidth=0.5)
        
        # Add dimension text
        plt.text(x_pos, -r_inner[i] - 0.02, f"r_i = {r_inner[i]*100:.2f} cm", 
                 ha='center', va='top', fontsize=8)
        plt.text(x_pos, r_outer[i] + 0.02, f"r_o = {r_outer[i]*100:.2f} cm", 
                 ha='center', va='bottom', fontsize=8)
    
    # Add arrow for force at the end
    plt.arrow(L, 0, 0, -0.1, head_width=0.05, head_length=0.05, 
              fc='red', ec='red', linewidth=2)
    plt.text(L+0.05, -0.05, f"F = {F/1000:.0f} kN", 
             color='red', fontsize=12, ha='left', va='center')
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title('Optimized Beam Design', fontsize=14, fontweight='bold')
    plt.xlabel('Length (m)', fontsize=12)
    plt.ylabel('Width (m)', fontsize=12)
    
    # Set equal aspect ratio and limits
    plt.axis('equal')
    max_radius = max(r_outer) * 1.5
    plt.xlim(-0.2, L+0.2)
    plt.ylim(-max_radius, max_radius)
    
    # Get the script's directory and save the plot there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_constraint_utilization(params, filename="constraint_utilization.png"):
    """Plot a bar chart showing the utilization percentage of each constraint."""
    # Calculate constraint values
    displacement_util = abs(calculate_displacement(params)) / max_displacement * 100
    
    # Calculate stress utilization for each segment
    r_outer = params[0:num_segments] / 100
    r_inner = params[num_segments:2*num_segments] / 100
    
    stress_utils = []
    for i in range(num_segments):
        # Calculate moment at this segment
        x = (i + 0.5) * segment_length
        moment = F * (L - x)
        
        # Calculate stress using beam bending formula
        I = calculate_area_moment_of_inertia(r_outer[i], r_inner[i])
        c = r_outer[i]  # Distance from neutral axis to the outer fiber
        stress = moment * c / I
        
        # Calculate utilization percentage
        stress_util = abs(stress) / sigma_yield * 100
        stress_utils.append(stress_util)
    
    # Calculate radii constraint utilization
    # For each segment, how close is inner radius to outer radius
    r_outer = params[0:num_segments]
    r_inner = params[num_segments:2*num_segments]
    
    radii_utils = []
    for i in range(num_segments):
        # If r_inner approaches r_outer, utilization approaches 100%
        radii_util = (r_inner[i] / r_outer[i]) * 100
        radii_utils.append(radii_util)
    
    # Calculate welding constraint utilization
    # For adjacent segments, how close is inner radius of segment i to outer radius of segment i+1
    welding_utils = []
    for i in range(num_segments-1):
        # If r_inner[i] approaches r_outer[i+1], utilization approaches 100%
        if r_outer[i+1] > r_inner[i]:
            welding_util = (r_inner[i] / r_outer[i+1]) * 100
        else:
            welding_util = 100  # Constraint violated
        welding_utils.append(welding_util)
    
    # Create figure for bar chart
    plt.figure(figsize=(12, 8))
    
    # Plot displacement constraint utilization
    plt.subplot(2, 2, 1)
    plt.bar(['Displacement'], [displacement_util], color='blue', alpha=0.7)
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.7)
    plt.title('Displacement Constraint Utilization', fontsize=12)
    plt.ylabel('Utilization (%)')
    plt.ylim(0, 110)
    for i, v in enumerate([displacement_util]):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=10)
    
    # Plot stress constraint utilization
    plt.subplot(2, 2, 2)
    segment_labels = [f"Segment {i+1}" for i in range(num_segments)]
    plt.bar(segment_labels, stress_utils, color='green', alpha=0.7)
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.7)
    plt.title('Stress Constraint Utilization', fontsize=12)
    plt.ylabel('Utilization (%)')
    plt.ylim(0, 110)
    plt.xticks(rotation=45)
    for i, v in enumerate(stress_utils):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=9)
    
    # Plot radii constraint utilization
    plt.subplot(2, 2, 3)
    plt.bar(segment_labels, radii_utils, color='orange', alpha=0.7)
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.7)
    plt.title('Radii Constraint Utilization (r_inner/r_outer)', fontsize=12)
    plt.ylabel('Utilization (%)')
    plt.ylim(0, 110)
    plt.xticks(rotation=45)
    for i, v in enumerate(radii_utils):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=9)
    
    # Plot welding constraint utilization
    plt.subplot(2, 2, 4)
    welding_labels = [f"Seg {i+1}-{i+2}" for i in range(num_segments-1)]
    plt.bar(welding_labels, welding_utils, color='purple', alpha=0.7)
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.7)
    plt.title('Welding Constraint Utilization', fontsize=12)
    plt.ylabel('Utilization (%)')
    plt.ylim(0, 110)
    plt.xticks(rotation=45)
    for i, v in enumerate(welding_utils):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Get the script's directory and save the plot there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimize a cantilever beam design')
    parser.add_argument('--force', type=float, default=500000, help='Force applied at the end of the beam (N)')
    parser.add_argument('--iterations', type=int, default=5000, help='Number of iterations for simulated annealing')
    parser.add_argument('--initial-temp', type=float, default=100.0, help='Initial temperature for simulated annealing')
    parser.add_argument('--cooling-rate', type=float, default=0.99, help='Cooling rate for simulated annealing')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    args = parser.parse_args()
    
    # Update force value if specified
    global F
    F = args.force
    
    # Initial parameters (in cm) as given in the problem
    initial_params = np.array([
        20, 19, 18, 17, 16,  # Outer radii
        10, 9, 8, 7, 6       # Inner radii
    ])
    
    print("Testing with initial parameters:")
    weight = objective_function(initial_params)
    displacement = calculate_displacement(initial_params)
    print(f"Weight: {weight:.2f} kg")
    print(f"Displacement: {abs(displacement)*100:.2f} cm")
    
    # Check constraints
    print(f"Displacement constraint: {constraint_displacement(initial_params) >= 0}")
    print(f"Radii constraint: {all(constraint_radii(initial_params) >= 0)}")
    print(f"Welding constraint: {all(constraint_welding(initial_params) >= 0)}")
    print(f"Stress constraint: {constraint_yield_stress(initial_params) >= 0}")
    
    # Run optimization
    print("\nStarting optimization...")
    best_params, best_weight, history = simulated_annealing(
        initial_params, 
        n_iterations=args.iterations, 
        initial_temp=args.initial_temp, 
        cooling_rate=args.cooling_rate
    )
    
    print("\nOptimization Results:")
    print(f"Best Weight: {best_weight:.2f} kg")
    
    # Extract optimized radii
    r_outer = best_params[0:num_segments]
    r_inner = best_params[num_segments:2*num_segments]
    
    print("\nOptimized Radii (cm):")
    for i in range(num_segments):
        print(f"Segment {i+1}: Outer Radius = {r_outer[i]:.2f}, Inner Radius = {r_inner[i]:.2f}")
    
    # Calculate final displacement
    final_displacement = calculate_displacement(best_params)
    print(f"Final Displacement: {abs(final_displacement)*100:.2f} cm")
    
    # Print constraints for the optimal solution
    print("\nConstraint Values for Optimal Solution:")
    print(f"Displacement constraint margin: {constraint_displacement(best_params):.2e} m")
    print(f"Stress constraint margin: {constraint_yield_stress(best_params)/1e6:.2f} MPa")
    
    # Save optimization results to a file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, 'optimization_results.txt')
    
    with open(results_path, 'w') as f:
        f.write(f"Optimization Results:\n")
        f.write(f"Best Weight: {best_weight:.2f} kg\n\n")
        f.write(f"Optimized Radii (cm):\n")
        for i in range(num_segments):
            f.write(f"Segment {i+1}: Outer Radius = {r_outer[i]:.2f}, Inner Radius = {r_inner[i]:.2f}\n")
        f.write(f"\nFinal Displacement: {abs(final_displacement)*100:.2f} cm\n")
        f.write(f"\nConstraint Values for Optimal Solution:\n")
        f.write(f"Displacement constraint margin: {constraint_displacement(best_params):.2e} m\n")
        f.write(f"Stress constraint margin: {constraint_yield_stress(best_params)/1e6:.2f} MPa\n")
    
    if not args.no_plot:
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        plt.plot(history)
        plt.xlabel('Iteration')
        plt.ylabel('Weight (kg)')
        plt.title('Optimization History')
        plt.grid(True)
        
        # Save history plot to script directory
        history_path = os.path.join(script_dir, 'optimization_history.png')
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        
        # Plot beam with optimized cross sections using the new function
        plot_optimized_beam(best_params)
        
        # Plot deformed shape of the beam
        displacements = plot_deformed_beam(best_params)
        
        # Plot constraint utilization
        plot_constraint_utilization(best_params)
        
        #plt.show()

if __name__ == "__main__":
    main() 