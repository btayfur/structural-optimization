import numpy as np
from scipy.spatial.distance import pdist, euclidean

def calculate_hypervolume(objectives, reference_point=None):
    """
    Calculate the hypervolume indicator for a set of objective values.
    
    Parameters:
    - objectives: Array of objective values (should be minimized)
    - reference_point: Reference point for hypervolume calculation
                       If None, use maximum values + 10% margin
    
    Returns:
    - Hypervolume value
    """
    if len(objectives) == 0:
        return 0.0
        
    # If no reference point is given, use the maximum values in each dimension + margin
    if reference_point is None:
        max_values = np.max(objectives, axis=0)
        reference_point = max_values * 1.1  # Add 10% margin
    
    # Sort points by first objective for the sweep line algorithm
    sorted_indices = np.argsort(objectives[:, 0])
    sorted_objectives = objectives[sorted_indices]
    
    # Initialize hypervolume
    hv = 0
    
    # Calculate hypervolume using the sweep line algorithm (for 2 objectives)
    for i in range(len(sorted_objectives)):
        if i == 0:
            # First point
            width = reference_point[0] - sorted_objectives[i, 0]
            height = reference_point[1] - sorted_objectives[i, 1]
        else:
            # Other points
            width = sorted_objectives[i-1, 0] - sorted_objectives[i, 0]
            height = reference_point[1] - sorted_objectives[i, 1]
        
        if width > 0 and height > 0:
            hv += width * height
    
    return hv

def calculate_igd(objectives, reference_front):
    """
    Calculate the Inverted Generational Distance (IGD) metric.
    
    Parameters:
    - objectives: Array of objective values from the algorithm
    - reference_front: Array of objective values representing the true Pareto front
    
    Returns:
    - IGD value (smaller is better)
    """
    if len(objectives) == 0 or len(reference_front) == 0:
        return float('inf')
    
    # Calculate the minimum distance from each point in the reference front
    # to any point in the approximated front
    total_distance = 0
    for ref_point in reference_front:
        min_dist = float('inf')
        for point in objectives:
            dist = euclidean(ref_point, point)
            if dist < min_dist:
                min_dist = dist
        total_distance += min_dist
    
    # Calculate the average distance
    igd = total_distance / len(reference_front)
    return igd

def calculate_spread(objectives):
    """
    Calculate the spread/diversity metric for a Pareto front approximation.
    
    Parameters:
    - objectives: Array of objective values
    
    Returns:
    - Spread value (smaller is better)
    """
    if len(objectives) < 2:
        return float('inf')
    
    # Sort by first objective
    sorted_indices = np.argsort(objectives[:, 0])
    sorted_points = objectives[sorted_indices]
    
    # Calculate distances between adjacent points
    distances = []
    for i in range(len(sorted_points) - 1):
        distances.append(euclidean(sorted_points[i], sorted_points[i+1]))
    
    if not distances:  # Boş kontrol
        return float('inf')
        
    distances = np.array(distances)
    
    # Calculate mean distance
    mean_distance = np.mean(distances)
    
    # Calculate the spread metric
    d_extremes = euclidean(sorted_points[0], sorted_points[-1])
    numerator = d_extremes + np.sum(np.abs(distances - mean_distance))
    denominator = d_extremes + (len(sorted_points) - 1) * mean_distance
    
    spread = numerator / denominator if denominator != 0 else float('inf')
    return spread 