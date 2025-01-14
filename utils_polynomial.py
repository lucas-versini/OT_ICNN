import sympy as sp
import numpy as np
from SumOfSquares import SOSProblem
from scipy.optimize import minimize
import scipy

x, y = sp.symbols('x y')

def is_sos(p):
    prob = SOSProblem()
    prob.add_sos_constraint(p, [x,])
    try:
        prob.solve()
        return True
    except:
        return False

def minimize_sos_polynomial(objective, d):
    """
    Minimize a polynomial of degree d subject to the constraint that it is a sum of squares

    Parameters
    ----------
    objective : callable
        Objective function to minimize
    d : int
        Degree of the polynomial

    Returns
    -------
    coeffs : array
        Coefficients of the polynomial
    fun_poly : callable
        Polynomial function
    """
    initial_coeffs = np.random.randn(d + 1)
    constraints = [{'type': 'ineq', 'fun': lambda P_coeff: 1 if is_sos(sum([P_coeff[i] * x**i for i in range(d+1)])) else -1}]
    result = minimize(fun = objective,
                      x0 = initial_coeffs,
                      constraints = constraints)
    
    P_coeff = sp.symbols(f'a0:{d + 1}')
    P_func = sp.lambdify((x, *P_coeff), sum([P_coeff[i] * x**i for i in range(d+1)]), 'numpy')
    fun_poly = lambda x: P_func(x, *result.x)

    return result.x, fun_poly

def aux(i):
    return ((i - 1) * i) // 2

def is_sos_2D(p):
    prob = SOSProblem()
    prob.add_sos_constraint(p, [x,y])
    try:
        prob.solve()
        return True
    except:
        return False

def minimize_sos_polynomial_2D(objective, d):
    """
    Minimize a polynomial of degree d subject to the constraint that it is a sum of squares

    Parameters
    ----------
    objective : callable
        Objective function to minimize
    d : int
        Degree of the polynomial
    
    Returns
    -------
    coeffs : array
        Coefficients of the polynomial
    fun_poly : callable
        Polynomial function
    """
    x, y = sp.symbols('x y')

    def P_coeff_to_P(P_coeff):
        return sum([sum([P_coeff[aux(i) + j] * x**i * y**j for j in range(d+1) if i+j <= d]) for i in range(d+1)])

    constraints = [{'type': 'ineq', 'fun': lambda P_coeff: 1 if is_sos_2D(P_coeff_to_P(P_coeff)) else -1}]
    result = minimize(fun = objective,
                        x0 = np.random.randn(int(scipy.special.comb(d+2, 2))),
                        constraints = constraints)

    P_coeff = sp.symbols([[f'P_{i}{j}' for j in range(d+1) if i + j <= d] for i in range(d+1)])
    P_func = sp.lambdify((x, y, *P_coeff), sum([sum([P_coeff[i][j] * x**i * y**j for j in range(d+1) if i+j <= d]) for i in range(d+1)]), 'numpy')
    fun_poly = lambda x, y: P_func(x, y, *result.x)

    return result.x, fun_poly
