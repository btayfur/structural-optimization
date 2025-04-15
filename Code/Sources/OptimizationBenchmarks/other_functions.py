"""
Other Benchmark Functions

This module contains additional benchmark test functions that don't fit neatly
into the previous categories.
"""

import numpy as np


def beale(x):
    """
    Beale Function
    
    f(x) = (1.5 - x1 + x1*x2)^2 + (2.25 - x1 + x1*x2^2)^2 + (2.625 - x1 + x1*x2^3)^2
    
    Global minimum: f(3, 0.5) = 0
    Bounds: [-4.5, 4.5]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Beale function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    term1 = (1.5 - x1 + x1*x2)**2
    term2 = (2.25 - x1 + x1*x2**2)**2
    term3 = (2.625 - x1 + x1*x2**3)**2
    
    return term1 + term2 + term3


def branin(x):
    """
    Branin Function (Branin-Hoo)
    
    f(x) = a*(x2 - b*x1^2 + c*x1 - d)^2 + e*(1-f)*cos(x1) + e
    where a=1, b=5.1/(4*pi^2), c=5/pi, d=6, e=10, f=1/(8*pi)
    
    Global minima: f(-pi, 12.275) = f(pi, 2.275) = f(9.42478, 2.475) = 0.397887
    Bounds: x1 in [-5, 10], x2 in [0, 15]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Branin function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    d = 6
    e = 10
    f = 1 / (8 * np.pi)
    
    term1 = a * (x2 - b * x1**2 + c * x1 - d)**2
    term2 = e * (1 - f) * np.cos(x1)
    
    return term1 + term2 + e


def colville(x):
    """
    Colville Function
    
    f(x) = 100(x1^2 - x2)^2 + (x1 - 1)^2 + (x3 - 1)^2 + 90(x3^2 - x4)^2 +
           10.1((x2 - 1)^2 + (x4 - 1)^2) + 19.8(x2 - 1)(x4 - 1)
    
    Global minimum: f(1, 1, 1, 1) = 0
    Bounds: [-10, 10]
    
    Parameters:
        x (array): Input vector [x1, x2, x3, x4]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 4:
        raise ValueError("Colville function is only defined for 4 dimensions")
    
    x1, x2, x3, x4 = x
    
    term1 = 100 * (x1**2 - x2)**2
    term2 = (x1 - 1)**2
    term3 = (x3 - 1)**2
    term4 = 90 * (x3**2 - x4)**2
    term5 = 10.1 * ((x2 - 1)**2 + (x4 - 1)**2)
    term6 = 19.8 * (x2 - 1) * (x4 - 1)
    
    return term1 + term2 + term3 + term4 + term5 + term6


def forrester(x):
    """
    Forrester et al. (2008) Function
    
    f(x) = (6*x - 2)^2 * sin(12*x - 4)
    
    Global minimum: f(0.75) ≈ -6.02074
    Bounds: [0, 1]
    
    Parameters:
        x (array): Input vector with a single element
    
    Returns:
        float: Function value at x
    """
    if len(x) != 1:
        raise ValueError("Forrester function is only defined for 1 dimension")
    
    x = x[0]
    
    return (6*x - 2)**2 * np.sin(12*x - 4)


def goldstein_price(x):
    """
    Goldstein-Price Function
    
    f(x) = [1 + (x1 + x2 + 1)^2 * (19 - 14*x1 + 3*x1^2 - 14*x2 + 6*x1*x2 + 3*x2^2)] *
           [30 + (2*x1 - 3*x2)^2 * (18 - 32*x1 + 12*x1^2 + 48*x2 - 36*x1*x2 + 27*x2^2)]
    
    Global minimum: f(0, -1) = 3
    Bounds: [-2, 2]
    
    Parameters:
        x (array): Input vector [x1, x2]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 2:
        raise ValueError("Goldstein-Price function is only defined for 2 dimensions")
    
    x1, x2 = x
    
    factor1a = (x1 + x2 + 1)**2
    factor1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
    factor1 = 1 + factor1a * factor1b
    
    factor2a = (2*x1 - 3*x2)**2
    factor2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
    factor2 = 30 + factor2a * factor2b
    
    return factor1 * factor2


def hartmann_3d(x):
    """
    Hartmann 3-D Function
    
    f(x) = -sum(alpha_i * exp(-sum(A_ij * (x_j - P_ij)^2)))
    
    Global minimum: f(0.114614, 0.555649, 0.852547) ≈ -3.86278
    Bounds: [0, 1]
    
    Parameters:
        x (array): Input vector [x1, x2, x3]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 3:
        raise ValueError("Hartmann 3D function is only defined for 3 dimensions")
        
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    
    A = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0]
    ])
    
    P = np.array([
        [0.3689, 0.1170, 0.2673],
        [0.4699, 0.4387, 0.7470],
        [0.1091, 0.8732, 0.5547],
        [0.0381, 0.5743, 0.8828]
    ])
    
    outer_sum = 0
    for i in range(4):
        inner_sum = 0
        for j in range(3):
            inner_sum += A[i, j] * (x[j] - P[i, j])**2
        outer_sum += alpha[i] * np.exp(-inner_sum)
    
    return -outer_sum


def hartmann_4d(x):
    """
    Hartmann 4-D Function
    
    f(x) = -sum(alpha_i * exp(-sum(A_ij * (x_j - P_ij)^2)))
    
    Global minimum: f(0.1873, 0.1906, 0.5566, 0.2647) ≈ -3.93416
    Bounds: [0, 1]
    
    Parameters:
        x (array): Input vector [x1, x2, x3, x4]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 4:
        raise ValueError("Hartmann 4D function is only defined for 4 dimensions")
        
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    
    A = np.array([
        [10.0, 3.0, 17.0, 3.5],
        [0.05, 10.0, 17.0, 0.1],
        [3.0, 3.5, 1.7, 10.0],
        [17.0, 8.0, 0.05, 10.0]
    ])
    
    P = np.array([
        [0.1312, 0.1696, 0.5569, 0.0124],
        [0.2329, 0.4135, 0.8307, 0.3736],
        [0.2348, 0.1415, 0.3522, 0.2883],
        [0.4047, 0.8828, 0.8732, 0.5743]
    ])
    
    outer_sum = 0
    for i in range(4):
        inner_sum = 0
        for j in range(4):
            inner_sum += A[i, j] * (x[j] - P[i, j])**2
        outer_sum += alpha[i] * np.exp(-inner_sum)
    
    return -outer_sum


def hartmann_6d(x):
    """
    Hartmann 6-D Function
    
    f(x) = -sum(alpha_i * exp(-sum(A_ij * (x_j - P_ij)^2)))
    
    Global minimum: f(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573) ≈ -3.32237
    Bounds: [0, 1]
    
    Parameters:
        x (array): Input vector [x1, x2, x3, x4, x5, x6]
    
    Returns:
        float: Function value at x
    """
    if len(x) != 6:
        raise ValueError("Hartmann 6D function is only defined for 6 dimensions")
        
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    
    A = np.array([
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
    ])
    
    P = np.array([
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
    ])
    
    outer_sum = 0
    for i in range(4):
        inner_sum = 0
        for j in range(6):
            inner_sum += A[i, j] * (x[j] - P[i, j])**2
        outer_sum += alpha[i] * np.exp(-inner_sum)
    
    return -outer_sum


def perm_function_db(x, beta=10):
    """
    Perm Function d, β
    
    f(x) = sum( (sum( (j^i + beta) * ((x_j/j)^i - 1) ))^2 )
    
    Global minimum: f(1, 2, 3, ..., d) = 0
    Bounds: [-d, d]
    
    Parameters:
        x (array): Input vector
        beta (float): Parameter that influences the difficulty of the problem
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    result = 0
    for i in range(1, n+1):
        sum_term = 0
        for j in range(1, n+1):
            sum_term += (j**i + beta) * ((x[j-1]/j)**i - 1)
        result += sum_term**2
        
    return result


def powell(x):
    """
    Powell Function
    
    f(x) = sum((x_{4i-3} + 10*x_{4i-2})^2 + 5*(x_{4i-1} - x_{4i})^2 +
                (x_{4i-2} - 2*x_{4i-1})^4 + 10*(x_{4i-3} - x_{4i})^4)
    
    Global minimum: f(0, 0, ..., 0) = 0
    Bounds: [-4, 5]
    
    Parameters:
        x (array): Input vector (dimension should be a multiple of 4)
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    n = len(x)
    
    if n % 4 != 0:
        raise ValueError("Powell function requires dimension to be a multiple of 4")
    
    result = 0
    for i in range(1, n//4 + 1):
        i4 = 4 * i
        term1 = (x[i4-4] + 10*x[i4-3])**2
        term2 = 5 * (x[i4-2] - x[i4-1])**2
        term3 = (x[i4-3] - 2*x[i4-2])**4
        term4 = 10 * (x[i4-4] - x[i4-1])**4
        result += term1 + term2 + term3 + term4
        
    return result


def shekel(x, m=10, a=None, c=None):
    """
    Shekel Function
    
    f(x) = -sum(1 / (sum((x_j - a_ij)^2) + c_i))
    
    Global minimum depends on m:
    For m=5: f(4, 4, 4, 4) ≈ -10.1532
    For m=7: f(4, 4, 4, 4) ≈ -10.4029
    For m=10: f(4, 4, 4, 4) ≈ -10.5364
    Bounds: [0, 10]
    
    Parameters:
        x (array): Input vector [x1, x2, x3, x4]
        m (int): Number of summation terms (typically 5, 7, or 10)
        a (array): Matrix of parameters, default is specific values
        c (array): Vector of parameters, default is specific values
    
    Returns:
        float: Function value at x
    """
    if len(x) != 4:
        raise ValueError("Shekel function is only defined for 4 dimensions")
    
    if a is None:
        a = np.array([
            [4.0, 4.0, 4.0, 4.0],
            [1.0, 1.0, 1.0, 1.0],
            [8.0, 8.0, 8.0, 8.0],
            [6.0, 6.0, 6.0, 6.0],
            [3.0, 7.0, 3.0, 7.0],
            [2.0, 9.0, 2.0, 9.0],
            [5.0, 5.0, 3.0, 3.0],
            [8.0, 1.0, 8.0, 1.0],
            [6.0, 2.0, 6.0, 2.0],
            [7.0, 3.6, 7.0, 3.6]
        ])
    
    if c is None:
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    
    result = 0
    for i in range(m):
        inner_sum = np.sum((x - a[i])**2) + c[i]
        result -= 1.0 / inner_sum
        
    return result


def styblinski_tang(x):
    """
    Styblinski-Tang Function
    
    f(x) = 0.5 * sum(x_i^4 - 16*x_i^2 + 5*x_i)
    
    Global minimum: f(-2.903534, ..., -2.903534) ≈ -39.16599*n
    Bounds: [-5, 5]
    
    Parameters:
        x (array): Input vector
    
    Returns:
        float: Function value at x
    """
    x = np.array(x)
    
    return 0.5 * np.sum(x**4 - 16*x**2 + 5*x) 