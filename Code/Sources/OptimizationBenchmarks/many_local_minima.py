"""
Benchmark Functions with Many Local Minima

This module contains test functions that have many local minima,
making them challenging for optimization algorithms.
"""

import numpy as np


def ackley(x):
    """
    Ackley Function
    
    f(x) = -20*exp(-0.2*sqrt(1/n*sum(x_i^2))) - exp(1/n*sum(cos(2*pi*x_i))) + 20 + e
    
    Global minimum: f(0,...,0) = 0
    Bounds: [-32.768, 32.768]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    
    return term1 + term2 + 20 + np.e


def bukin_n6(x):
    """
    Bukin Function N. 6
    
    f(x) = 100 * sqrt(|x2 - 0.01*x1^2|) + 0.01*|x1 + 10|
    
    Global minimum: f(-10, 1) = 0
    Bounds: x1 in [-15, -5], x2 in [-3, 3]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Bukin N.6 function is only defined for 2 dimensions")
    
    x1, x2 = x
    term1 = 100 * np.sqrt(abs(x2 - 0.01 * x1**2))
    term2 = 0.01 * abs(x1 + 10)
    
    return term1 + term2


def cross_in_tray(x):
    """
    Cross-in-Tray Function
    
    f(x) = -0.0001 * (|sin(x1) * sin(x2) * exp(|100 - sqrt(x1^2 + x2^2)/pi|)| + 1)^0.1
    
    Global minima: 
    f(±1.3491, ±1.3491) = -2.06261
    Bounds: [-10, 10]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Cross-in-tray function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    fact1 = np.sin(x1) * np.sin(x2)
    fact2 = np.exp(abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
    
    return -0.0001 * (abs(fact1 * fact2) + 1)**0.1


def drop_wave(x):
    """
    Drop-Wave Function
    
    f(x) = -(1 + cos(12 * sqrt(x1^2 + x2^2))) / (0.5 * (x1^2 + x2^2) + 2)
    
    Global minimum: f(0, 0) = -1
    Bounds: [-5.12, 5.12]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Drop-wave function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    frac1 = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
    frac2 = 0.5 * (x1**2 + x2**2) + 2
    
    return -frac1 / frac2


def eggholder(x):
    """
    Eggholder Function
    
    f(x) = -(x2 + 47) * sin(sqrt(|x2 + x1/2 + 47|)) - x1 * sin(sqrt(|x1 - (x2 + 47)|))
    
    Global minimum: f(512, 404.2319) = -959.6407
    Bounds: [-512, 512]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Eggholder function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    term1 = -(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1/2 + 47)))
    term2 = -x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))
    
    return term1 + term2


def gramacy_lee(x):
    """
    Gramacy & Lee (2012) Function
    
    f(x) = sin(10*pi*x) / (2*x) + (x-1)^4
    
    Global minimum: f(0.548) = -0.869
    Bounds: [0.5, 2.5]
    
    Parameters:
        x (array): Input vector with a single element
    
    Returns:
        float: Function value at x
    """
    if len(x) != 1:
        raise ValueError("Gramacy & Lee function is only defined for 1 dimension")
    
    x = x[0]
    
    term1 = np.sin(10 * np.pi * x) / (2 * x)
    term2 = (x - 1)**4
    
    return term1 + term2


def griewank(x):
    """
    Griewank Function
    
    f(x) = 1 + sum(x_i^2 / 4000) - prod(cos(x_i / sqrt(i)))
    
    Global minimum: f(0,...,0) = 0
    Bounds: [-600, 600]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    sum_term = np.sum(x**2) / 4000
    
    # Using a loop for the product term to handle the division by sqrt(i)
    prod_term = 1
    for i in range(n):
        prod_term *= np.cos(x[i] / np.sqrt(i + 1))
    
    return 1 + sum_term - prod_term


def holder_table(x):
    """
    Holder Table Function
    
    f(x) = -|sin(x1) * cos(x2) * exp(|1 - sqrt(x1^2 + x2^2)/pi|)|
    
    Global minima: f(±8.05502, ±9.66459) = -19.2085
    Bounds: [-10, 10]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Holder table function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    term = np.sin(x1) * np.cos(x2) * np.exp(abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))
    
    return -abs(term)


def langermann(x, m=5, c=None, A=None):
    """
    Langermann Function
    
    f(x) = sum(c_i * exp(-1/pi * sum((x_j - A_ij)^2)) * cos(pi * sum((x_j - A_ij)^2)))
    
    Parameters:
        x (array): Input vector
        m (int): Number of summation terms
        c (array): Coefficients, default is [1, 2, 5, 2, 3]
        A (array): Matrix of parameters, default is specific values
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    if c is None:
        c = np.array([1, 2, 5, 2, 3])
    
    if A is None:
        if n == 2:
            # Default values for 2D
            A = np.array([
                [3, 5], [5, 2], [2, 1], [1, 4], [7, 9]
            ])
        else:
            raise ValueError("For dimensions other than 2, matrix A must be provided")
    
    result = 0
    for i in range(m):
        sum_sq = np.sum((x - A[i])**2)
        result += c[i] * np.exp(-sum_sq / np.pi) * np.cos(np.pi * sum_sq)
    
    return result


def levy(x):
    """
    Levy Function
    
    For d dimensions:
    f(x) = sin^2(pi*w_1) + sum((w_i-1)^2 * (1+10*sin^2(pi*w_i+1))) + (w_d-1)^2 * (1+sin^2(2*pi*w_d))
    where w_i = 1 + (x_i - 1)/4
    
    Global minimum: f(1,...,1) = 0
    Bounds: [-10, 10]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    d = len(x)
    
    w = 1 + (x - 1) / 4
    
    term1 = np.sin(np.pi * w[0])**2
    
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    
    return term1 + term2 + term3


def levy_n13(x):
    """
    Levy Function N. 13
    
    f(x) = sin^2(3*pi*x1) + (x1-1)^2 * (1+sin^2(3*pi*x2)) + (x2-1)^2 * (1+sin^2(2*pi*x2))
    
    Global minimum: f(1, 1) = 0
    Bounds: [-10, 10]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Levy N.13 function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    term1 = np.sin(3 * np.pi * x1)**2
    term2 = (x1 - 1)**2 * (1 + np.sin(3 * np.pi * x2)**2)
    term3 = (x2 - 1)**2 * (1 + np.sin(2 * np.pi * x2)**2)
    
    return term1 + term2 + term3


def rastrigin(x):
    """
    Rastrigin Function
    
    f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Global minimum: f(0,...,0) = 0
    Bounds: [-5.12, 5.12]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def schaffer_n2(x):
    """
    Schaffer Function N. 2
    
    f(x) = 0.5 + (sin^2(x1^2 - x2^2) - 0.5) / (1 + 0.001*(x1^2 + x2^2))^2
    
    Global minimum: f(0, 0) = 0
    Bounds: [-100, 100]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Schaffer N.2 function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    numerator = np.sin(x1**2 - x2**2)**2 - 0.5
    denominator = (1 + 0.001 * (x1**2 + x2**2))**2
    
    return 0.5 + numerator / denominator


def schaffer_n4(x):
    """
    Schaffer Function N. 4
    
    f(x) = 0.5 + (cos^2(sin(|x1^2 - x2^2|)) - 0.5) / (1 + 0.001*(x1^2 + x2^2))^2
    
    Global minimum: f(0, ±1.25313) = 0.292579
    Bounds: [-100, 100]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Schaffer N.4 function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    numerator = np.cos(np.sin(abs(x1**2 - x2**2)))**2 - 0.5
    denominator = (1 + 0.001 * (x1**2 + x2**2))**2
    
    return 0.5 + numerator / denominator


def schwefel(x):
    """
    Schwefel Function
    
    f(x) = 418.9829*n - sum(x_i*sin(sqrt(|x_i|)))
    
    Global minimum: f(420.9687,...,420.9687) = 0
    Bounds: [-500, 500]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def shubert(x):
    """
    Shubert Function
    
    f(x) = (sum_{i=1}^5 i*cos((i+1)*x1 + i)) * (sum_{i=1}^5 i*cos((i+1)*x2 + i))
    
    Multiple global minima, value: -186.7309
    Bounds: [-10, 10]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Shubert function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    sum1 = sum(i * np.cos((i + 1) * x1 + i) for i in range(1, 6))
    sum2 = sum(i * np.cos((i + 1) * x2 + i) for i in range(1, 6))
    
    return sum1 * sum2 