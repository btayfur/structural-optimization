"""
Steep Ridges/Drops Benchmark Functions

This module contains test functions that have steep ridges, drops, or narrow peaks,
making them challenging for gradient-based optimization algorithms.
"""

import numpy as np


def de_jong_n5(x, a=None):
    """
    De Jong Function N. 5 (Shekel's Foxholes)
    
    f(x) = (0.002 + sum(1 / (j + sum((x_i - a_ij)^6))))^(-1)
    
    Global minimum: f(-32, -32) ≈ 0.998
    Bounds: [-65.536, 65.536]
    
    Parameters:
        x (array): Input vector [x1, x2]
        a (array): Matrix of parameters, default is specific values
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("De Jong N.5 function is only defined for 2 dimensions")
    
    if a is None:
        # Default parameter values
        a = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                a.append([32 * i, 32 * j])
        a = np.array(a)
    
    sum_term = 0
    for j in range(25):
        inner_sum = 0
        for i in range(2):
            inner_sum += (x[i] - a[j, i])**6
        sum_term += 1.0 / (j + 1 + inner_sum)
    
    return 1.0 / (0.002 + sum_term)


def easom(x):
    """
    Easom Function
    
    f(x) = -cos(x1) * cos(x2) * exp(-(x1-pi)^2 - (x2-pi)^2)
    
    Global minimum: f(pi, pi) = -1
    Bounds: [-100, 100]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Easom function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    cos_term = np.cos(x1) * np.cos(x2)
    exp_term = np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))
    
    return -cos_term * exp_term


def michalewicz(x, m=10):
    """
    Michalewicz Function
    
    f(x) = -sum(sin(x_i) * sin(i*x_i^2/pi)^(2*m))
    
    Global minimum depends on dimension:
    For d=2: f(?) ≈ -1.8013
    For d=5: f(?) ≈ -4.6877
    For d=10: f(?) ≈ -9.6602
    Bounds: [0, pi]
    
    Parameters:
        x (array): Input vector
        m (int): Steepness parameter (typically m=10)
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    result = 0
    for i in range(n):
        result -= np.sin(x[i]) * np.sin(((i+1) * x[i]**2) / np.pi)**(2*m)
        
    return result 