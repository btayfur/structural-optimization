"""
Plate-Shaped Benchmark Functions

This module contains plate-shaped test functions that have relatively flat
regions near the minimum, which can make them challenging for some optimization algorithms.
"""

import numpy as np


def booth(x):
    """
    Booth Function
    
    f(x) = (x1 + 2*x2 - 7)^2 + (2*x1 + x2 - 5)^2
    
    Global minimum: f(1, 3) = 0
    Bounds: [-10, 10]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Booth function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    term1 = (x1 + 2*x2 - 7)**2
    term2 = (2*x1 + x2 - 5)**2
    
    return term1 + term2


def matyas(x):
    """
    Matyas Function
    
    f(x) = 0.26 * (x1^2 + x2^2) - 0.48 * x1 * x2
    
    Global minimum: f(0, 0) = 0
    Bounds: [-10, 10]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Matyas function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


def mccormick(x):
    """
    McCormick Function
    
    f(x) = sin(x1 + x2) + (x1 - x2)^2 - 1.5*x1 + 2.5*x2 + 1
    
    Global minimum: f(-0.54719, -1.54719) = -1.9133
    Bounds: x1 in [-1.5, 4], x2 in [-3, 4]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("McCormick function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    term1 = np.sin(x1 + x2)
    term2 = (x1 - x2)**2
    term3 = -1.5*x1 + 2.5*x2 + 1
    
    return term1 + term2 + term3


def power_sum(x, b=None):
    """
    Power Sum Function
    
    f(x) = sum( ( sum( x_j^i ) - b_i )^2 )
    
    Global minimum depends on the b values
    Default b values for dimension 4: [8, 18, 44, 114]
    Bounds: [0, d], where d is the dimension
    
    Parameters:
        x (array): Input vector
        b (array): Target values for each power sum
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    if b is None:
        if n == 4:
            b = [8, 18, 44, 114]
        else:
            raise ValueError("For dimensions other than 4, the b array must be provided")
    elif len(b) != n:
        raise ValueError("The length of b must match the dimension of x")
    
    result = 0
    for i in range(1, n+1):
        inner_sum = np.sum(x**i)
        result += (inner_sum - b[i-1])**2
        
    return result


def zakharov(x):
    """
    Zakharov Function
    
    f(x) = sum(x_i^2) + (sum(0.5*i*x_i))^2 + (sum(0.5*i*x_i))^4
    
    Global minimum: f(0,...,0) = 0
    Bounds: [-5, 10]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    sum1 = np.sum(x**2)
    
    weighted_sum = np.sum([0.5*i*x[i-1] for i in range(1, n+1)])
    sum2 = weighted_sum**2
    sum3 = weighted_sum**4
    
    return sum1 + sum2 + sum3 