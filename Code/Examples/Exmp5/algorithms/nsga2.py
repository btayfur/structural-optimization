import numpy as np
from utils.problem import binh_and_korn

def nsga2(pop_size, max_gen, crossover_rate, mutation_rate, bounds):
    """
    NSGA-II algorithm for multi-objective optimization.
    
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
    # Initialize population
    dim = len(bounds)
    population = initialize_population(pop_size, dim, bounds)
    
    # Evaluate initial population
    objectives = np.array([binh_and_korn(ind) for ind in population])
    
    # Main evolution loop
    for generation in range(max_gen):
        # Create offspring population through selection, crossover, and mutation
        offspring_population = create_offspring(population, objectives, bounds, crossover_rate, mutation_rate)
        
        # Evaluate offspring
        offspring_objectives = np.array([binh_and_korn(ind) for ind in offspring_population])
        
        # Combine parent and offspring populations
        combined_population = np.vstack((population, offspring_population))
        combined_objectives = np.vstack((objectives, offspring_objectives))
        
        # Fast non-dominated sort
        fronts = fast_non_dominated_sort(combined_objectives)
        
        # Calculate crowding distance
        crowding_distances = []
        for front in fronts:
            cd = calculate_crowding_distance(combined_objectives[front])
            crowding_distances.append(cd)
        
        # Select the next generation
        population, objectives = select_next_generation(combined_population, combined_objectives, fronts, crowding_distances, pop_size)
    
    # Get the first Pareto front in the final population
    fronts = fast_non_dominated_sort(objectives)
    first_front = fronts[0]
    
    return population[first_front], objectives[first_front]

def initialize_population(pop_size, dim, bounds):
    """Initialize population with random values within bounds"""
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        population[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], pop_size)
    return population

def create_offspring(population, objectives, bounds, crossover_rate, mutation_rate):
    """Create offspring population through selection, crossover, and mutation"""
    offspring_population = []
    
    while len(offspring_population) < len(population):
        # Binary tournament selection
        parent1_idx = binary_tournament_selection(objectives)
        parent2_idx = binary_tournament_selection(objectives)
        
        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]
        
        # Simulated Binary Crossover (SBX)
        if np.random.rand() < crossover_rate:
            child1, child2 = sbx_crossover(parent1, parent2, bounds)
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        # Polynomial Mutation
        child1 = polynomial_mutation(child1, bounds, mutation_rate)
        child2 = polynomial_mutation(child2, bounds, mutation_rate)
        
        offspring_population.append(child1)
        if len(offspring_population) < len(population):
            offspring_population.append(child2)
    
    return np.array(offspring_population)

def binary_tournament_selection(objectives):
    """Binary tournament selection based on dominance and crowding distance"""
    idx1, idx2 = np.random.randint(0, len(objectives), 2)
    
    # Check if one dominates the other
    if dominates(objectives[idx1], objectives[idx2]):
        return idx1
    if dominates(objectives[idx2], objectives[idx1]):
        return idx2
    
    # If no domination, select the one with larger crowding distance (or randomly if equal)
    return np.random.choice([idx1, idx2])

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

def dominates(obj1, obj2):
    """Check if obj1 dominates obj2"""
    return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

def fast_non_dominated_sort(objectives):
    """
    Fast non-dominated sort algorithm from NSGA-II.
    Returns a list of fronts, where each front is a list of indices.
    """
    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)  # Number of solutions that dominate solution i
    dominated_solutions = [[] for _ in range(n)]  # List of solutions that solution i dominates
    
    # Calculate domination counts and dominated solutions
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                elif dominates(objectives[j], objectives[i]):
                    domination_count[i] += 1
    
    # Identify all fronts
    fronts = []
    current_front = np.where(domination_count == 0)[0]
    
    while len(current_front) > 0:
        fronts.append(current_front)
        next_front = []
        
        for i in current_front:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        
        current_front = np.array(next_front)
    
    return fronts

def calculate_crowding_distance(objectives):
    """
    Calculate crowding distance for a set of objective values.
    Returns an array of crowding distances.
    """
    n = len(objectives)
    if n <= 2:
        return np.ones(n) * np.inf
    
    # Initialize crowding distance
    crowding_distance = np.zeros(n)
    
    # For each objective
    for obj_idx in range(objectives.shape[1]):
        # Sort by this objective
        sorted_idx = np.argsort(objectives[:, obj_idx])
        sorted_objectives = objectives[sorted_idx]
        
        # Set boundary points to infinity
        crowding_distance[sorted_idx[0]] = np.inf
        crowding_distance[sorted_idx[-1]] = np.inf
        
        # Calculate crowding distance for other points
        obj_range = sorted_objectives[-1, obj_idx] - sorted_objectives[0, obj_idx]
        if obj_range > 0:  # Avoid division by zero
            for i in range(1, n-1):
                crowding_distance[sorted_idx[i]] += (sorted_objectives[i+1, obj_idx] - sorted_objectives[i-1, obj_idx]) / obj_range
    
    return crowding_distance

def select_next_generation(population, objectives, fronts, crowding_distances, pop_size):
    """
    Select the next generation based on non-dominated sorting and crowding distance.
    """
    next_population = []
    next_objectives = []
    
    # Add solutions from each front until we reach pop_size
    front_idx = 0
    while len(next_population) + len(fronts[front_idx]) <= pop_size and front_idx < len(fronts):
        # Add all solutions from this front
        for idx in fronts[front_idx]:
            next_population.append(population[idx])
            next_objectives.append(objectives[idx])
        front_idx += 1
    
    # If we need more solutions to fill up pop_size
    if len(next_population) < pop_size and front_idx < len(fronts):
        # Sort the current front by crowding distance
        last_front = fronts[front_idx]
        last_front_distances = crowding_distances[front_idx]
        sorted_idx = np.argsort(-last_front_distances)  # Note the negative sign for descending order
        
        # Add solutions up to pop_size
        remaining = pop_size - len(next_population)
        for i in range(min(remaining, len(sorted_idx))):
            idx = last_front[sorted_idx[i]]
            next_population.append(population[idx])
            next_objectives.append(objectives[idx])
    
    return np.array(next_population), np.array(next_objectives) 