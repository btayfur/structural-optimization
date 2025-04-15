"""
Bowl-Shaped Benchmark Functions

This module contains bowl-shaped test functions that typically have a
single minimum surrounded by circular or elliptical contours.
"""

import numpy as np


def bohachevsky(x, version=1):
    """
    Bohachevsky Functions
    
    Three variants of this function:
    f1(x) = x1^2 + 2*x2^2 - 0.3*cos(3*pi*x1) - 0.4*cos(4*pi*x2) + 0.7
    f2(x) = x1^2 + 2*x2^2 - 0.3*cos(3*pi*x1)*cos(4*pi*x2) + 0.3
    f3(x) = x1^2 + 2*x2^2 - 0.3*cos(3*pi*x1 + 4*pi*x2) + 0.3
    
    Global minimum: f(0, 0) = 0
    Bounds: [-100, 100]
    
    Parameters:
        x (array): Input vector [x1, x2]
        version (int): Which variant to use (1, 2, or 3)
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Bohachevsky function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    if version == 1:
        return x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2) + 0.7
    elif version == 2:
        return x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1)*np.cos(4*np.pi*x2) + 0.3
    elif version == 3:
        return x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1 + 4*np.pi*x2) + 0.3
    else:
        raise ValueError("Version must be 1, 2, or 3")


def perm_0(x, beta=0.5):
    """
    Perm Function 0, d, β
    
    f(x) = sum( sum( (j + beta) * (x_i^j - (1/i)^j) )^2 )
    
    Global minimum: f(1, 1/2, 1/3, ..., 1/n) = 0
    Bounds: [-n, n]
    
    Parameters:
        x (array): Input vector
        beta (float): Parameter that influences the difficulty of the problem
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    outer_sum = 0
    
    for i in range(1, n+1):
        inner_sum = 0
        for j in range(1, n+1):
            inner_sum += (j + beta) * (x[j-1]**i - (1/j)**i)
        outer_sum += inner_sum**2
        
    return outer_sum


def rotated_hyper_ellipsoid(x):
    """
    Rotated Hyper-Ellipsoid Function
    
    f(x) = sum( sum( x_j^2 ) )
    
    Global minimum: f(0,...,0) = 0
    Bounds: [-65.536, 65.536]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    result = 0
    for i in range(n):
        for j in range(i+1):
            result += x[j]**2
            
    return result


def sphere(x):
    """
    Sphere Function
    
    f(x) = sum(x_i^2)
    
    Global minimum: f(0,...,0) = 0
    Bounds: [-5.12, 5.12]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    return np.sum(x**2)


def sum_of_different_powers(x):
    """
    Sum of Different Powers Function
    
    f(x) = sum(|x_i|^(i+1))
    
    Global minimum: f(0,...,0) = 0
    Bounds: [-1, 1]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    return sum(abs(x[i])**(i+2) for i in range(n))


def sum_squares(x):
    """
    Sum Squares Function
    
    f(x) = sum(i*x_i^2)
    
    Global minimum: f(0,...,0) = 0
    Bounds: [-10, 10]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    return sum((i+1) * x[i]**2 for i in range(n))


def trid(x):
    """
    Trid Function
    
    f(x) = sum((x_i - 1)^2) - sum(x_i * x_{i-1})
    
    Global minimum: f(x*) = -n(n+4)(n-1)/6, where n is the dimension
    and x*_i = i(n+1-i), i=1,2,...,n
    Bounds: [-n^2, n^2]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    sum1 = np.sum((x - 1)**2)
    sum2 = np.sum(x[1:] * x[:-1])
    
    return sum1 - sum2 