import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from algorithms.weighted_sum import weighted_sum
from algorithms.nsga2 import nsga2
from algorithms.spea2 import spea2
from algorithms.moead import moead
from utils.metrics import calculate_hypervolume, calculate_igd, calculate_spread
from utils.problem import binh_and_korn, plot_pareto_front, get_true_pareto_front
import os

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Problem parameters - Set smaller values for faster execution
pop_size = 50  # Reduced from 100
max_gen = 30   # Reduced from 100
crossover_rate = 0.9
mutation_rate = 0.1
bounds = [(0, 5), (0, 3)]  # Bounds for Binh and Korn problem
objective_weights = [0.5, 0.5]  # For weighted sum method

# Run all algorithms
print("Running Weighted Sum...")
start_time = time.time()
ws_solutions, ws_objectives = weighted_sum(
    pop_size, max_gen, crossover_rate, mutation_rate, bounds, objective_weights
)
ws_time = time.time() - start_time
print(f"Weighted Sum completed in {ws_time:.2f} seconds")

print("\nRunning NSGA-II...")
start_time = time.time()
nsga2_solutions, nsga2_objectives = nsga2(
    pop_size, max_gen, crossover_rate, mutation_rate, bounds
)
nsga2_time = time.time() - start_time
print(f"NSGA-II completed in {nsga2_time:.2f} seconds")

print("\nRunning SPEA2...")
start_time = time.time()
spea2_solutions, spea2_objectives = spea2(
    pop_size, max_gen, crossover_rate, mutation_rate, bounds
)
spea2_time = time.time() - start_time
print(f"SPEA2 completed in {spea2_time:.2f} seconds")

print("\nRunning MOEA/D...")
start_time = time.time()
moead_solutions, moead_objectives = moead(
    pop_size, max_gen, crossover_rate, mutation_rate, bounds
)
moead_time = time.time() - start_time
print(f"MOEA/D completed in {moead_time:.2f} seconds")

# Calculate metrics
print("\nCalculating performance metrics...")

# Get approximate true Pareto front for reference
reference_front = get_true_pareto_front(num_points=100)

# Calculate hypervolume
ws_hv = calculate_hypervolume(ws_objectives)
nsga2_hv = calculate_hypervolume(nsga2_objectives)
spea2_hv = calculate_hypervolume(spea2_objectives)
moead_hv = calculate_hypervolume(moead_objectives)

# Calculate IGD (Inverted Generational Distance)
ws_igd = calculate_igd(ws_objectives, reference_front)
nsga2_igd = calculate_igd(nsga2_objectives, reference_front)
spea2_igd = calculate_igd(spea2_objectives, reference_front)
moead_igd = calculate_igd(moead_objectives, reference_front)

# Calculate spread (diversity)
ws_spread = calculate_spread(ws_objectives)
nsga2_spread = calculate_spread(nsga2_objectives)
spea2_spread = calculate_spread(spea2_objectives)
moead_spread = calculate_spread(moead_objectives)

# Print results table
print("\nPerformance Metrics:")
print("-" * 80)
print(f"{'Algorithm':<10} | {'Hypervolume':<15} | {'IGD':<15} | {'Spread':<15} | {'Time (s)':<10}")
print("-" * 80)
print(f"{'WS':<10} | {ws_hv:<15.4f} | {ws_igd:<15.4f} | {ws_spread:<15.4f} | {ws_time:<10.2f}")
print(f"{'NSGA-II':<10} | {nsga2_hv:<15.4f} | {nsga2_igd:<15.4f} | {nsga2_spread:<15.4f} | {nsga2_time:<10.2f}")
print(f"{'SPEA2':<10} | {spea2_hv:<15.4f} | {spea2_igd:<15.4f} | {spea2_spread:<15.4f} | {spea2_time:<10.2f}")
print(f"{'MOEA/D':<10} | {moead_hv:<15.4f} | {moead_igd:<15.4f} | {moead_spread:<15.4f} | {moead_time:<10.2f}")
print("-" * 80)

# Plot individual Pareto fronts first
def plot_individual_pareto(objectives, title, filename):
    plt.figure(figsize=(10, 8))
    plt.scatter(objectives[:, 0], objectives[:, 1], c='blue', marker='o', alpha=0.8)
    
    # Plot true Pareto front as a reference
    plt.scatter(reference_front[:, 0], reference_front[:, 1], c='red', marker='.', alpha=0.3, s=10, label='True Pareto Front')
    
    plt.title(title)
    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{filename}_pareto.png')
    plt.close()

plot_individual_pareto(ws_objectives, 'Weighted Sum Pareto Front', 'ws')
plot_individual_pareto(nsga2_objectives, 'NSGA-II Pareto Front', 'nsga2')
plot_individual_pareto(spea2_objectives, 'SPEA2 Pareto Front', 'spea2')
plot_individual_pareto(moead_objectives, 'MOEA/D Pareto Front', 'moead')

# Plot all Pareto fronts together for comparison
plt.figure(figsize=(12, 10))
plt.scatter(ws_objectives[:, 0], ws_objectives[:, 1], label='Weighted Sum', alpha=0.6)
plt.scatter(nsga2_objectives[:, 0], nsga2_objectives[:, 1], label='NSGA-II', alpha=0.6)
plt.scatter(spea2_objectives[:, 0], spea2_objectives[:, 1], label='SPEA2', alpha=0.6)
plt.scatter(moead_objectives[:, 0], moead_objectives[:, 1], label='MOEA/D', alpha=0.6)
plt.scatter(reference_front[:, 0], reference_front[:, 1], c='red', marker='.', alpha=0.3, s=5, label='True Pareto Front')
plt.title('Comparison of Pareto Fronts for Binh and Korn Function')
plt.xlabel('f1(x)')
plt.ylabel('f2(x)')
plt.legend()
plt.grid(True)
plt.savefig('results/pareto_comparison.png')
plt.close()

# Save individual algorithm results with 3D plot (solutions + objective space)
plot_pareto_front(ws_solutions, ws_objectives, 'Weighted Sum', 
                 'results/weighted_sum_3d.png')
plot_pareto_front(nsga2_solutions, nsga2_objectives, 'NSGA-II',
                 'results/nsga2_3d.png')
plot_pareto_front(spea2_solutions, spea2_objectives, 'SPEA2',
                 'results/spea2_3d.png')
plot_pareto_front(moead_solutions, moead_objectives, 'MOEA/D',
                 'results/moead_3d.png')

# Plot metrics comparison as a bar chart
plt.figure(figsize=(14, 8))

# Set up data for plotting
algorithms = ['Weighted Sum', 'NSGA-II', 'SPEA2', 'MOEA/D']
hypervolumes = [ws_hv, nsga2_hv, spea2_hv, moead_hv]
igds = [ws_igd, nsga2_igd, spea2_igd, moead_igd]
spreads = [ws_spread, nsga2_spread, spea2_spread, moead_spread]
times = [ws_time, nsga2_time, spea2_time, moead_time]

# Set positions for bars
x = np.arange(len(algorithms))
width = 0.2

plt.subplot(2, 2, 1)
plt.bar(x, hypervolumes, width, label='Hypervolume (higher is better)')
plt.ylabel('Hypervolume')
plt.title('Hypervolume Comparison')
plt.xticks(x, algorithms, rotation=45)

plt.subplot(2, 2, 2)
plt.bar(x, igds, width, label='IGD (lower is better)')
plt.ylabel('IGD')
plt.title('Inverted Generational Distance Comparison')
plt.xticks(x, algorithms, rotation=45)

plt.subplot(2, 2, 3)
plt.bar(x, spreads, width, label='Spread (lower is better)')
plt.ylabel('Spread')
plt.title('Spread/Diversity Comparison')
plt.xticks(x, algorithms, rotation=45)

plt.subplot(2, 2, 4)
plt.bar(x, times, width, label='Computation Time (seconds)')
plt.ylabel('Time (s)')
plt.title('Computation Time Comparison')
plt.xticks(x, algorithms, rotation=45)

plt.tight_layout()
plt.savefig('results/metrics_comparison.png')
plt.close()

print("\nAll results saved to the 'results' directory") 