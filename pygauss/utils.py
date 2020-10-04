#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:42:48 2019

@author: maximevono
"""

from math import cos,pi
import numpy as np

class Chebyshev:
    """
    This class computes a Chebyshev approximation of a given function.
    """
    def __init__(self, a, b, n, func):
        """
        Parameters
        ----------
        a : float
            Lower bound of the interval [a,b] on which the function is
            approximated.
        b : float
            Upper bound of the interval [a,b] on which the function is
            approximated.
        n : int
            Degree of the polynomial approximation.
        func : function
            Function to be approximated.
        """
        self.a = a
        self.b = b
        self.func = func

        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)
        f = [func(cos(pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
        fac = 2.0 / n
        self.c = [fac * sum([f[k] * cos(pi * j * (k + 0.5) / n)
                  for k in range(n)]) for j in range(n)]

    def eval(self, x):
        """
        Parameters
        ----------
        x : float
            Point at which the approximation has to be computed.
            
        Returns
        -------
        The value of the approximation at x.
        """
        a,b = self.a, self.b
        assert(a <= x <= b)
        y = (2.0 * x - a - b) * (1.0 / (b - a))
        y2 = 2.0 * y
        (d, dd) = (self.c[-1], 0)             # Special case first step for efficiency
        for cj in self.c[-2:0:-1]:            # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * self.c[0]   # Last step is different


def col_vector_norms(a,order=None):
    """
    """
    if isinstance(a,np.matrix):
        a = a.A
    norms = np.fromiter((np.linalg.norm(col,order) for col in a.T),a.dtype)
    return norms



def CG(A,b,x=None,K=None,tol=None,info=False):
    """
    """
    
    # Default parameters
    if x is None:
        x = np.zeros(b.shape)
    if K is None:
        K = np.size(b,0)
    if tol is None:
        tol = 1e-4
    
    # Initialization
    k = 1
    r_old = b - A(x)
    p_old = r_old.copy()
    d_old = (p_old * A(p_old)).sum(axis=0)
    r_new = np.ones(b.shape)
    
    # CG algorithm
    while ((col_vector_norms(r_new,2) >= tol).all() and k <= K):
        gam = (r_old * r_old).sum(axis=0) / d_old
        x[:] += p_old * gam
        r_new[:] = r_old - A(p_old) * gam
        beta = - (r_new * r_new).sum(axis=0) / (r_old * r_old).sum(axis=0)
        p_new = r_new - p_old * beta
        d_new = (p_new * A(p_new)).sum(axis=0)
        r_old = r_new
        p_old = p_new
        d_old = d_new
        k = k + 1
    
    # Output
    if info:
        return (x, k-1, col_vector_norms(r_new,2))
    else:
        return x
    
    
def diagonal_form(A,lower=0,upper=0):
    """
    A is a numpy square matrix
    this function converts a square matrix to diagonal ordered form
    returned matrix in ab shape which can be used directly for scipy.linalg.solve_banded
    """
    n = A.shape[1]
    assert(np.all(A.shape ==(n,n)))
    
    ab = np.zeros((2*n-1, n))
    
    for i in range(n):
        ab[i,(n-1)-i:] = np.diagonal(A,(n-1)-i)
        
    for i in range(n-1): 
        ab[(2*n-2)-i,:i+1] = np.diagonal(A,i-(n-1))

    mid_row_inx = int(ab.shape[0]/2)
    upper_rows = [mid_row_inx - i for i in range(1, upper+1)]
    upper_rows.reverse()
    upper_rows.append(mid_row_inx)
    lower_rows = [mid_row_inx + i for i in range(1, lower+1)]
    keep_rows = upper_rows+lower_rows
    ab = ab[keep_rows,:]

    return ab


def triangular_inversion(triang_arg):
    """Counts inversion of a triangular matrix (lower or upper).

    Args:
        triang_arg (np.matrix, np.array): triangular matrix for inversion

    Returns:
        np.matrix: inverse of triangular matrix
        
    Raises:
        Exception: An error occured while passing non-square matrix
        Exception: An error occured while passing non-triangular matrix
        Exception: An error occured while passing singular matrix

    """

    triang = np.matrix(triang_arg.copy())
    n = triang.shape[0]

    unitriang_maker = np.matrix(np.identity(n)) / triang.diagonal()
    unitriang = unitriang_maker * triang
    nilpotent = unitriang - np.identity(n)

    unitriang_inverse = np.matrix(np.identity(n))
    for i in range(n-1):
        unitriang_inverse = np.matrix(np.identity(n)) - nilpotent * unitriang_inverse

    return unitriang_inverse * unitriang_maker