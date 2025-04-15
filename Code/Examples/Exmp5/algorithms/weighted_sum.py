import numpy as np
from utils.problem import binh_and_korn

def weighted_sum(pop_size, max_gen, crossover_rate, mutation_rate, bounds, weights):
    """
    Weighted Sum Method for multi-objective optimization.
    
    Parameters:
    - pop_size: Population size
    - max_gen: Maximum number of generations
    - crossover_rate: Probability of crossover
    - mutation_rate: Probability of mutation
    - bounds: List of tuples (min, max) for each decision variable
    - weights: List of weights for each objective function
    
    Returns:
    - Best solutions found (in decision space)
    - Corresponding objective values
    """
    # Run multiple times with different weight combinations
    # to better approximate the Pareto front
    num_weight_vectors = 10  # Number of different weight vectors to use
    all_solutions = []
    all_objectives = []
    
    for w_idx in range(num_weight_vectors):
        # Create different weight vectors
        if num_weight_vectors > 1:
            weights_i = np.array([w_idx / (num_weight_vectors - 1), 1 - w_idx / (num_weight_vectors - 1)])
        else:
            weights_i = np.array(weights)
        
        # Normalize weights
        weights_i = weights_i / np.sum(weights_i)
        
        # Initialize population
        dim = len(bounds)
        population = initialize_population(pop_size, dim, bounds)
        
        # Evaluate initial population
        objectives = np.array([binh_and_korn(ind) for ind in population])
        
        # Main evolution loop
        for generation in range(max_gen):
            # Calculate the weighted sum for each solution
            weighted_objectives = np.sum(objectives * weights_i, axis=1)
            
            # Create new population
            new_population = []
            
            # Elitism: Keep the best solution
            best_idx = np.argmin(weighted_objectives)
            new_population.append(population[best_idx])
            
            # Generate the rest of the population
            while len(new_population) < pop_size:
                # Selection
                parent1_idx = tournament_selection(weighted_objectives)
                parent2_idx = tournament_selection(weighted_objectives)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # Crossover
                if np.random.rand() < crossover_rate:
                    child1, child2 = sbx_crossover(parent1, parent2, bounds)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = polynomial_mutation(child1, bounds, mutation_rate)
                child2 = polynomial_mutation(child2, bounds, mutation_rate)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < pop_size:
                    new_population.append(child2)
            
            # Update population
            population = np.array(new_population)
            
            # Re-evaluate population
            objectives = np.array([binh_and_korn(ind) for ind in population])
        
        # Get the final best solution for this weight vector
        best_idx = np.argmin(np.sum(objectives * weights_i, axis=1))
        all_solutions.append(population[best_idx])
        all_objectives.append(objectives[best_idx])
    
    # Combine results from all weight vectors
    all_solutions = np.array(all_solutions)
    all_objectives = np.array(all_objectives)
    
    # Extract non-dominated solutions
    non_dominated_indices = find_non_dominated(all_objectives)
    pareto_solutions = all_solutions[non_dominated_indices]
    pareto_objectives = all_objectives[non_dominated_indices]
    
    return pareto_solutions, pareto_objectives

def initialize_population(pop_size, dim, bounds):
    """Initialize population with random values within bounds"""
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        population[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], pop_size)
    return population

def tournament_selection(weighted_objectives, tournament_size=2):
    """Select an individual using tournament selection (lower is better)"""
    indices = np.random.randint(0, len(weighted_objectives), tournament_size)
    return indices[np.argmin(weighted_objectives[indices])]

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