import numpy as np
from scipy.spatial.distance import cdist
from utils.problem import binh_and_korn

def spea2(pop_size, max_gen, crossover_rate, mutation_rate, bounds):
    """
    SPEA2 algorithm for multi-objective optimization.
    
    Parameters:
    - pop_size: Population size
    - max_gen: Maximum number of generations
    - crossover_rate: Probability of crossover
    - mutation_rate: Probability of mutation
    - bounds: List of tuples (min, max) for each decision variable
    
    Returns:
    - Best solutions found (in decision space)
    - Corresponding objective values
    """
    # Size of the archive
    archive_size = pop_size
    
    # Initialize population
    dim = len(bounds)
    population = initialize_population(pop_size, dim, bounds)
    
    # Initialize archive
    archive = np.zeros((0, dim))
    archive_obj = np.zeros((0, 2))  # For 2 objectives
    
    # Evaluate initial population
    objectives = np.array([binh_and_korn(ind) for ind in population])
    
    # Main evolution loop
    for generation in range(max_gen):
        # Combine population and archive
        if len(archive) > 0:
            combined_pop = np.vstack((population, archive))
            combined_obj = np.vstack((objectives, archive_obj))
        else:
            combined_pop = population
            combined_obj = objectives
        
        # Calculate fitness values for the combined population
        fitness = calculate_fitness(combined_obj)
        
        # Environmental selection
        archive, archive_obj, archive_fitness = environmental_selection(combined_pop, combined_obj, fitness, archive_size)
        
        # Check if max generations reached
        if generation == max_gen - 1:
            break
        
        # Mating selection
        mating_pool = binary_tournament_selection(archive, archive_fitness, pop_size)
        
        # Variation (crossover and mutation)
        offspring = []
        for i in range(0, len(mating_pool), 2):
            if i + 1 < len(mating_pool):
                parent1 = mating_pool[i]
                parent2 = mating_pool[i + 1]
                
                # Apply crossover
                if np.random.rand() < crossover_rate:
                    child1, child2 = sbx_crossover(parent1, parent2, bounds)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Apply mutation
                child1 = polynomial_mutation(child1, bounds, mutation_rate)
                child2 = polynomial_mutation(child2, bounds, mutation_rate)
                
                offspring.append(child1)
                offspring.append(child2)
        
        # Update population and objectives
        if len(offspring) > 0:
            population = np.array(offspring)
            objectives = np.array([binh_and_korn(ind) for ind in population])
    
    # Return the final archive
    return archive, archive_obj

def initialize_population(pop_size, dim, bounds):
    """Initialize population with random values within bounds"""
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        population[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], pop_size)
    return population

def calculate_fitness(objectives):
    """
    Calculate SPEA2 fitness values for a population.
    
    Returns:
    - Fitness value for each individual (lower is better)
    """
    n = len(objectives)
    
    # Calculate dominance matrix
    dominance_matrix = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                dominance_matrix[i, j] = dominates(objectives[i], objectives[j])
    
    # Calculate raw fitness: number of solutions that dominate each solution
    strength = np.sum(dominance_matrix, axis=1)  # Number of solutions dominated by i
    raw_fitness = np.zeros(n)
    for i in range(n):
        indices = np.where(dominance_matrix[:, i])[0]  # Solutions that dominate i
        raw_fitness[i] = np.sum(strength[indices]) if len(indices) > 0 else 0
    
    # Calculate density
    distances = cdist(objectives, objectives)
    np.fill_diagonal(distances, np.inf)  # Ensure distance to self is infinity
    
    # Calculate k-th nearest neighbor distance (k = sqrt(n))
    k = min(int(np.sqrt(n)), n-1)  # Ensure k is within valid range
    k_distances = np.sort(distances, axis=1)[:, k] if n > 1 else np.zeros(n)
    
    # Calculate density (inverse of the k-th distance)
    density = 1.0 / (k_distances + 1e-10)  # Add small constant to avoid division by zero
    
    # Final fitness: raw fitness + density
    fitness = raw_fitness + density
    
    return fitness

def environmental_selection(population, objectives, fitness, archive_size):
    """
    SPEA2 environmental selection.
    
    Returns:
    - Selected archive
    - Objective values of the archive
    - Fitness values of the archive
    """
    if len(fitness) == 0:
        return np.zeros((0, population.shape[1])), np.zeros((0, objectives.shape[1])), np.zeros(0)
    
    # Find individuals with fitness < 1 (non-dominated)
    non_dominated_indices = np.where(fitness < 1)[0]
    
    if len(non_dominated_indices) == 0:
        # If no non-dominated solutions found, select best dominated solutions
        sorted_indices = np.argsort(fitness)
        selected_indices = sorted_indices[:min(archive_size, len(sorted_indices))]
    elif len(non_dominated_indices) == archive_size:
        selected_indices = non_dominated_indices
    elif len(non_dominated_indices) < archive_size:
        # If non-dominated set is smaller than archive size, add dominated solutions
        dominated_indices = np.where(fitness >= 1)[0]
        sorted_dominated_indices = dominated_indices[np.argsort(fitness[dominated_indices])]
        
        # Fill archive
        remaining = archive_size - len(non_dominated_indices)
        if len(sorted_dominated_indices) > 0:
            selected_indices = np.concatenate((non_dominated_indices, sorted_dominated_indices[:min(remaining, len(sorted_dominated_indices))]))
        else:
            selected_indices = non_dominated_indices
    else:
        # If non-dominated set is larger than archive size, truncate using clustering
        selected_indices = truncate_by_clustering(non_dominated_indices, objectives, archive_size)
    
    # Return selected archive
    return population[selected_indices], objectives[selected_indices], fitness[selected_indices]

def truncate_by_clustering(indices, objectives, archive_size):
    """
    Truncate a set of individuals based on clustering.
    
    Returns:
    - Indices of selected individuals
    """
    # Extract objectives of the individuals to be truncated
    obj_values = objectives[indices]
    
    # Calculate pairwise distances
    distances = cdist(obj_values, obj_values)
    np.fill_diagonal(distances, np.inf)  # Ensure distance to self is infinity
    
    # Iteratively remove individuals until we reach the desired size
    remaining_indices = np.arange(len(indices))
    
    while len(remaining_indices) > archive_size:
        # For each individual, find its nearest neighbor
        min_distances = np.min(distances, axis=1)
        
        # Find the individual with the smallest distance to its nearest neighbor
        min_dist_idx = np.argmin(min_distances)
        
        # Remove this individual
        distances = np.delete(distances, min_dist_idx, axis=0)
        distances = np.delete(distances, min_dist_idx, axis=1)
        remaining_indices = np.delete(remaining_indices, min_dist_idx)
    
    # Return the indices of the selected individuals
    return indices[remaining_indices]

def binary_tournament_selection(population, fitness, n_select):
    """
    Binary tournament selection for SPEA2.
    
    Returns:
    - Selected individuals
    """
    selected = []
    
    if len(population) == 0:
        return np.array(selected)
    
    while len(selected) < n_select:
        # Randomly select two individuals
        idx1, idx2 = np.random.randint(0, len(population), 2)
        
        # Select the one with better fitness (lower value is better in SPEA2)
        if fitness[idx1] < fitness[idx2]:
            selected.append(population[idx1])
        else:
            selected.append(population[idx2])
    
    return np.array(selected)

def dominates(obj1, obj2):
    """Check if obj1 dominates obj2"""
    return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

def sbx_crossover(parent1, parent2, bounds, eta=15):
    """Simulated Binary Crossover"""
    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent2)
    
    for i in range(len(parent1)):
        # Check if crossover happens for this gene
        if np.random.random() <= 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-10:  # Avoid division by zero
                if parent1[i] < parent2[i]:
                    y1, y2 = parent1[i], parent2[i]
                else:
                    y1, y2 = parent2[i], parent1[i]
                
                lower, upper = bounds[i]
                
                # Calculate beta
                rand = np.random.random()
                beta = 1.0 + (2.0 * (y1 - lower) / (y2 - y1))
                alpha = 2.0 - beta ** (-(eta + 1.0))
                
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                
                # Generate children
                c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                
                # Ensure the children are within bounds
                c1 = max(lower, min(upper, c1))
                c2 = max(lower, min(upper, c2))
                
                # Randomly assign to child1 and child2
                if np.random.random() <= 0.5:
                    child1[i], child2[i] = c1, c2
                else:
                    child1[i], child2[i] = c2, c1
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        else:
            child1[i] = parent1[i]
            child2[i] = parent2[i]
    
    return child1, child2

def polynomial_mutation(individual, bounds, mutation_rate, eta=20):
    """Polynomial Mutation"""
    child = individual.copy()
    
    for i in range(len(individual)):
        if np.random.random() <= mutation_rate:
            lower, upper = bounds[i]
            x = individual[i]
            delta1 = (x - lower) / (upper - lower)
            delta2 = (upper - x) / (upper - lower)
            
            rand = np.random.random()
            mut_pow = 1.0 / (eta + 1.0)
            
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                delta_q = 1.0 - val ** mut_pow
            
            x = x + delta_q * (upper - lower)
            x = max(lower, min(upper, x))
            child[i] = x
    
    return child 