import numpy as np
from scipy.spatial.distance import cdist
from utils.problem import binh_and_korn

def moead(pop_size, max_gen, crossover_rate, mutation_rate, bounds):
    """
    MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) implementation.
    
    Parameters:
    - pop_size: Population size (number of weight vectors)
    - max_gen: Maximum number of generations
    - crossover_rate: Probability of crossover
    - mutation_rate: Probability of mutation
    - bounds: List of tuples (min, max) for each decision variable
    
    Returns:
    - Best solutions found (in decision space)
    - Corresponding objective values
    """
    # Number of objectives (2 for Binh and Korn)
    m = 2
    
    # Number of neighbors (T)
    T = max(int(0.1 * pop_size), 2)
    
    # Initialize weight vectors (uniformly distributed)
    weights = initialize_weights(pop_size, m)
    
    # Compute the neighborhood of each weight vector
    B = compute_neighborhoods(weights, T)
    
    # Initialize population
    dim = len(bounds)
    population = initialize_population(pop_size, dim, bounds)
    
    # Evaluate initial population
    objectives = np.array([binh_and_korn(ind) for ind in population])
    
    # Initialize ideal point
    z_ideal = np.min(objectives, axis=0)
    
    # Initialize external population (archive)
    EP = []
    EP_objectives = []
    
    # Main loop
    for gen in range(max_gen):
        # For each subproblem
        for i in range(pop_size):
            # Select parents from neighborhood with probability 0.9
            # or from the entire population with probability 0.1
            if np.random.random() < 0.9:
                neighbors = B[i]
            else:
                neighbors = list(range(pop_size))
            
            # Select mating parents
            k, l = np.random.choice(neighbors, 2, replace=False) if len(neighbors) >= 2 else (0, 0)
            
            # Generate offspring
            if np.random.random() < crossover_rate and k != l:
                offspring = sbx_crossover_single(population[k], population[l], bounds)
            else:
                offspring = population[i].copy()
            
            # Apply mutation
            offspring = polynomial_mutation(offspring, bounds, mutation_rate)
            
            # Evaluate offspring
            offspring_obj = binh_and_korn(offspring)
            
            # Update ideal point
            z_ideal = np.minimum(z_ideal, offspring_obj)
            
            # Update neighbors
            for j in neighbors:
                # Calculate decomposed objective value using Tchebycheff approach
                gx_new = decompose_tchebycheff(offspring_obj, weights[j], z_ideal)
                gx_old = decompose_tchebycheff(objectives[j], weights[j], z_ideal)
                
                if gx_new <= gx_old:
                    population[j] = offspring.copy()
                    objectives[j] = offspring_obj.copy()
            
            # Update external population
            is_dominated = False
            to_remove = []
            
            for j, ep_obj in enumerate(EP_objectives):
                if dominates(offspring_obj, ep_obj):
                    to_remove.append(j)
                elif dominates(ep_obj, offspring_obj):
                    is_dominated = True
                    break
            
            # Remove dominated solutions from EP
            for j in reversed(to_remove):
                EP.pop(j)
                EP_objectives.pop(j)
            
            # Add new solution to EP if not dominated
            if not is_dominated and not any(np.array_equal(offspring_obj, ep_obj) for ep_obj in EP_objectives):
                EP.append(offspring.copy())
                EP_objectives.append(offspring_obj.copy())
    
    # Convert EP to numpy arrays
    if len(EP) > 0:
        EP = np.array(EP)
        EP_objectives = np.array(EP_objectives)
        return EP, EP_objectives
    else:
        # If EP is empty, return non-dominated solutions from population
        non_dominated_indices = find_non_dominated(objectives)
        if len(non_dominated_indices) > 0:
            return population[non_dominated_indices], objectives[non_dominated_indices]
        else:
            # If no non-dominated solutions found, return the best solution according to the first objective
            best_idx = np.argmin(objectives[:, 0])
            return population[best_idx:best_idx+1], objectives[best_idx:best_idx+1]

def initialize_weights(pop_size, m):
    """
    Initialize uniformly distributed weight vectors.
    For bi-objective problems, simply divide the range [0,1] into pop_size segments.
    
    Parameters:
    - pop_size: Population size
    - m: Number of objectives
    
    Returns:
    - weight_vectors: Array of weight vectors
    """
    if m == 2:
        # For bi-objective, simply create vectors of the form (i/(pop_size-1), 1-i/(pop_size-1))
        weights = np.zeros((pop_size, m))
        for i in range(pop_size):
            if pop_size > 1:
                weights[i, 0] = i / (pop_size - 1)
            else:
                weights[i, 0] = 0.5
            weights[i, 1] = 1 - weights[i, 0]
    else:
        # For more objectives, a more complex approach is needed (not implemented here)
        # This is a simplified approach for dimensions > 2
        weights = np.random.dirichlet(np.ones(m), pop_size)
        
    return weights

def compute_neighborhoods(weights, T):
    """
    Compute the neighborhood of each weight vector.
    
    Parameters:
    - weights: Array of weight vectors
    - T: Number of neighbors
    
    Returns:
    - neighborhoods: List of lists containing the indices of the T closest weight vectors
    """
    # Calculate Euclidean distances between weight vectors
    distances = cdist(weights, weights)
    
    # For each weight vector, find the T closest ones
    neighborhoods = []
    for i in range(len(weights)):
        # Argsort gives indices of elements in sorted order
        sorted_indices = np.argsort(distances[i])
        # Select the T closest (including itself)
        neighborhoods.append(sorted_indices[:min(T, len(sorted_indices))])
    
    return neighborhoods

def initialize_population(pop_size, dim, bounds):
    """Initialize population with random values within bounds"""
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        population[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], pop_size)
    return population

def sbx_crossover_single(parent1, parent2, bounds, eta=15):
    """Simulated Binary Crossover that returns a single offspring"""
    child = np.zeros_like(parent1)
    
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
                
                # Randomly select one of the children
                if np.random.random() <= 0.5:
                    child[i] = c1
                else:
                    child[i] = c2
            else:
                child[i] = parent1[i]
        else:
            child[i] = np.random.random() <= 0.5 and parent1[i] or parent2[i]
    
    return child

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

def decompose_tchebycheff(obj, weight, z_ideal):
    """
    Decompose a multi-objective problem using the Tchebycheff approach.
    
    Parameters:
    - obj: Objective values
    - weight: Weight vector
    - z_ideal: Ideal point
    
    Returns:
    - Scalar fitness value
    """
    return np.max(weight * np.abs(obj - z_ideal))

def dominates(obj1, obj2):
    """Check if obj1 dominates obj2"""
    return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

def find_non_dominated(objectives):
    """Find indices of non-dominated solutions"""
    n_points = objectives.shape[0]
    is_non_dominated = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_non_dominated[i]:
            # Check if point i dominates or is dominated by any other point
            for j in range(n_points):
                if i != j and is_non_dominated[j]:
                    # Check if j dominates i
                    if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                        is_non_dominated[i] = False
                        break
    
    return np.where(is_non_dominated)[0] 