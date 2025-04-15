import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Seaborn minimalist theme settings
sns.set_theme(style="white", font="sans-serif", font_scale=1.3)

# 1D Schwefel function
def schwefel_1d(x):
    """
    Schwefel Function in 1D
    
    f(x) = 418.9829 - x * sin(sqrt(|x|))
    
    Global minimum: f(420.9687) = 0
    Bounds: [-500, 500]
    """
    return 418.9829 - x * np.sin(np.sqrt(np.abs(x)))

# Color palette (professional, neutral)
main_color = "#1f2937"      # Dark gray-blue
accent_color = "#e11d48"    # Soft crimson
highlight_color = "#3b82f6" # Blue
constraint_color = "#f59e0b" # Amber
bounds_color = "#10b981"    # Emerald

# Create visualizations
def plot_objective_function():
    """Create a visualization of the Schwefel objective function in 1D"""
    x = np.linspace(-500, 500, 1000)
    y = schwefel_1d(x)
    
    # Find global minimum
    min_idx = np.argmin(y)
    min_x = x[min_idx]
    min_y = y[min_idx]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the function
    ax.plot(x, y, color=main_color, linewidth=2.5, label="Schwefel Function (1D)")
    
    # Mark the global minimum
    ax.scatter(min_x, min_y, color=accent_color, s=80, zorder=5)
    ax.axvline(x=min_x, color=accent_color, linestyle="--", linewidth=1.8, alpha=0.5)
    
    # Add annotation
    ax.annotate(f"Global Minimum\nx ≈ {min_x:.1f}", 
                xy=(min_x, min_y), 
                xytext=(min_x + 100, min_y + 50),
                arrowprops=dict(arrowstyle="->", color=accent_color),
                fontsize=12, color=accent_color)
    
    # Set title and labels
    ax.set_title("Objective Function: Schwefel (1D)", fontsize=18, fontweight='bold', loc='left', pad=20)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("f(x)", fontsize=14)
    
    # Remove spines and add grid
    sns.despine(trim=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Optimize margins
    plt.tight_layout()
    plt.savefig("objective_function.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_equality_constraints():
    """Create a visualization of equality constraints on the Schwefel function"""
    x = np.linspace(-500, 500, 1000)
    y = schwefel_1d(x)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the function
    ax.plot(x, y, color=main_color, linewidth=2.5, label="Schwefel Function (1D)")
    
    # Add equality constraints: x = -250 and x = 200
    constraint_x1 = -250
    constraint_x2 = 200
    constraint_y1 = schwefel_1d(constraint_x1)
    constraint_y2 = schwefel_1d(constraint_x2)
    
    # Plot constraint lines
    ax.axvline(x=constraint_x1, color=constraint_color, linestyle="-", linewidth=2)
    ax.axvline(x=constraint_x2, color=highlight_color, linestyle="-", linewidth=2)
    
    # Mark constraint points
    ax.scatter([constraint_x1, constraint_x2], [constraint_y1, constraint_y2], 
               color=[constraint_color, highlight_color], s=80, zorder=5)
    
    # Add annotations
    ax.annotate(f"Equality Constraint 1\nx = {constraint_x1}", 
                xy=(constraint_x1, constraint_y1), 
                xytext=(constraint_x1 - 150, constraint_y1 + 100),
                arrowprops=dict(arrowstyle="->", color=constraint_color),
                fontsize=12, color=constraint_color)
    ax.annotate(f"Equality Constraint 2\nx = {constraint_x2}", 
                xy=(constraint_x2, constraint_y2), 
                xytext=(constraint_x2 + 100, constraint_y2 + 100),
                arrowprops=dict(arrowstyle="->", color=highlight_color),
                fontsize=12, color=highlight_color)
    
    # Set title and labels
    ax.set_title("Equality Constraints: Schwefel (1D)", fontsize=18, fontweight='bold', loc='left', pad=20)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("f(x)", fontsize=14)
    
    # Remove spines and add grid
    sns.despine(trim=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Optimize margins
    plt.tight_layout()
    plt.savefig("equality_constraints.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_inequality_constraints():
    """Create a visualization of inequality constraints on the Schwefel function"""
    x = np.linspace(-500, 500, 1000)
    y = schwefel_1d(x)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the function
    ax.plot(x, y, color=main_color, linewidth=2.5, label="Schwefel Function (1D)")
    
    # Add inequality constraints
    # x <= 100 (first constraint region)
    constraint1_x = np.linspace(-500, 100, 500)
    constraint1_y = schwefel_1d(constraint1_x)
    
    # x >= -150 (second constraint region)
    constraint2_x = np.linspace(-150, 500, 500)
    constraint2_y = schwefel_1d(constraint2_x)
    
    # Create constraint regions (shared region is -150 <= x <= 100)
    ax.fill_between(constraint1_x, 0, 1000, color=constraint_color, alpha=0.2)
    ax.fill_between(constraint2_x, 0, 1000, color=highlight_color, alpha=0.2)
    
    # Add constraint boundary lines
    ax.axvline(x=100, color=constraint_color, linestyle="--", linewidth=2, 
               label="Inequality Constraint 1: x ≤ 100")
    ax.axvline(x=-150, color=highlight_color, linestyle="--", linewidth=2, 
               label="Inequality Constraint 2: x ≥ -150")
    
    # Highlight the feasible region (intersection)
    feasible_x = np.linspace(-150, 100, 200)
    ax.fill_between(feasible_x, 0, 1000, color="green", alpha=0.15, label="Feasable Region")
    
    # Add annotations
    ax.annotate("x ≤ 100", 
                xy=(100, schwefel_1d(100)), 
                xytext=(0, 200),
                arrowprops=dict(arrowstyle="->", color=constraint_color),
                fontsize=12, color=constraint_color)
    
    ax.annotate("x ≥ -150", 
                xy=(-150, schwefel_1d(-150)), 
                xytext=(-300, 200),
                arrowprops=dict(arrowstyle="->", color=highlight_color),
                fontsize=12, color=highlight_color)
    
    ax.annotate("Feasable Region\n-150 ≤ x ≤ 100", 
                xy=(-25, schwefel_1d(-25)), 
                xytext=(-25, 600),
                ha='center',
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=12, color="green")
    
    # Set title and labels
    ax.set_title("Inequality Constraints: Schwefel (1D)", fontsize=18, fontweight='bold', loc='left', pad=20)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("f(x)", fontsize=14)
    ax.set_ylim(0, 800)
    
    # Remove spines and add grid
    sns.despine(trim=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Optimize margins
    plt.tight_layout()
    plt.savefig("inequality_constraints.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_boundary_constraints():
    """Create a visualization of boundary constraints on the Schwefel function"""
    # Use a wider range to show boundary effects
    x = np.linspace(-600, 600, 1000)
    y = schwefel_1d(x)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the function
    ax.plot(x, y, color=main_color, linewidth=2.5, label="Schwefel Function (1D)")
    
    # Define boundary constraints
    lower_bound = -300
    upper_bound = 400
    
    # Highlight feasible region
    ax.axvspan(lower_bound, upper_bound, alpha=0.2, color=bounds_color)
    
    # Add boundary lines
    ax.axvline(x=lower_bound, color=bounds_color, linestyle="-", linewidth=2)
    ax.axvline(x=upper_bound, color=bounds_color, linestyle="-", linewidth=2)
    
    # Add annotations
    ax.annotate(f"Lower Bound\nx = {lower_bound}", 
                xy=(lower_bound, schwefel_1d(lower_bound)), 
                xytext=(lower_bound - 100, 600),
                arrowprops=dict(arrowstyle="->", color=bounds_color),
                fontsize=12, color=bounds_color)
    
    ax.annotate(f"Upper Bound\nx = {upper_bound}", 
                xy=(upper_bound, schwefel_1d(upper_bound)), 
                xytext=(upper_bound - 100, 600),
                arrowprops=dict(arrowstyle="->", color=bounds_color),
                fontsize=12, color=bounds_color)
    
    ax.text(50, 200, "Feasable Region\n-300 ≤ x ≤ 400", 
            fontsize=14, ha='center', color=bounds_color,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Set title and labels
    ax.set_title("Boundary Constraints: Schwefel (1D)", fontsize=18, fontweight='bold', loc='left', pad=20)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("f(x)", fontsize=14)
    
    # Remove spines and add grid
    sns.despine(trim=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Optimize margins
    plt.tight_layout()
    plt.savefig("boundary_constraints.png", dpi=300, bbox_inches="tight")
    plt.close()

# Generate all plots
if __name__ == "__main__":
    plot_objective_function()
    plot_equality_constraints()
    plot_inequality_constraints()
    plot_boundary_constraints()
    print("Tüm görseller oluşturuldu.")
