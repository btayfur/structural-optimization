"""
Valley-Shaped Benchmark Functions

This module contains valley-shaped test functions that typically have
long, narrow valleys that make them challenging for optimization algorithms.
"""

import numpy as np


def three_hump_camel(x):
    """
    Three-Hump Camel Function
    
    f(x) = 2*x1^2 - 1.05*x1^4 + (x1^6)/6 + x1*x2 + x2^2
    
    Global minimum: f(0, 0) = 0
    Bounds: [-5, 5]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Three-hump camel function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    term1 = 2*x1**2
    term2 = -1.05*x1**4
    term3 = (x1**6)/6
    term4 = x1*x2
    term5 = x2**2
    
    return term1 + term2 + term3 + term4 + term5


def six_hump_camel(x):
    """
    Six-Hump Camel Function
    
    f(x) = (4-2.1*x1^2 + (x1^4)/3)*x1^2 + x1*x2 + (-4+4*x2^2)*x2^2
    
    Global minima: f(±0.0898, ±0.7126) = -1.0316
    Bounds: x1 in [-3, 3], x2 in [-2, 2]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Six-hump camel function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    term1 = (4 - 2.1*x1**2 + (x1**4)/3)*x1**2
    term2 = x1*x2
    term3 = (-4 + 4*x2**2)*x2**2
    
    return term1 + term2 + term3


def dixon_price(x):
    """
    Dixon-Price Function
    
    f(x) = (x1 - 1)^2 + sum(i * (2*x_i^2 - x_{i-1})^2)
    
    Global minimum: f(x*) = 0, where x*_i = 2^(-(2^i - 2)/(2^i))
    Bounds: [-10, 10]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    term1 = (x[0] - 1)**2
    
    sum_term = 0
    for i in range(1, n):
        sum_term += (i+1) * (2*x[i]**2 - x[i-1])**2
        
    return term1 + sum_term


def rosenbrock(x):
    """
    Rosenbrock Function (Banana function)
    
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2)
    
    Global minimum: f(1,...,1) = 0
    Bounds: [-5, 10]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    if n < 2:
        raise ValueError("Rosenbrock function requires at least 2 dimensions")
    
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2) 