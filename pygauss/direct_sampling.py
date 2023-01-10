#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Implementation of direct approaches to sample from multivariate 
Gaussian distributions.

.. seealso::
    
    `Documentation on ReadTheDocs <https://pygauss-gaussian-sampling.readthedocs.io/en/latest/direct_sampling/index.html>`_
"""

import numpy as np
from scipy.linalg import solve_triangular,sqrtm
from math import sqrt
from pygauss.utils import CG, Chebyshev, col_vector_norms

#####################
# Special instances #
#####################

# Multivariate Gaussian sampling with band precision or covariance matrix
def sampler_band(mu,A,b,mode="precision",seed=None,size=1):
    r"""
    Algorithm dedicated to sample from a multivariate real-valued Gaussian 
    distribution :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{A})` or 
    :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{A}^{-1})` when 
    :math:`\mathbf{A}` is a band matrix.
    
    Parameters
    ----------
    mu : 1-D array_like, of length d
        Mean of the d-dimensional Gaussian distribution.
    A : 2-D array_like, of shape (d, d)
        Covariance or precision matrix of the distribution. It must be 
        symmetric and positive-definite for proper sampling.
    b : int
        Bandwidth of A.
    mode : string, optional
        Indicates if A refers to the precision or covariance matrix of the
        Gaussian distribution.
    seed : int, optional
               Random seed to reproduce experimental results.
    size : int, optional
        Given a size of for instance T, T independent and identically 
        distributed (i.i.d.) samples are returned.
        
    Returns
    -------
    theta : ndarray, of shape (d,size)
        The drawn samples, of shape (d,size), if that was provided. If not, 
        the shape is (d,1).
        
    Raises
    ------
    ValueError
        If mode is not included in ['covariance','precision'].
        
    Examples
    --------
    >>> d = 2
    >>> mu = np.zeros(d)
    >>> A = np.eye(2)
    >>> b = 0
    >>> mode = "covariance"
    >>> size = 1
    >>> theta = sampler_band(mu,A,b,mode=mode,seed=2022,size=size)
    """    

    # Set random seed
    np.random.seed(seed)      

    if mode == "precision":
        # Check if the matrix is definite positive
        if np.all(np.linalg.eigvals(A) > 0) == False:
            raise ValueError('''The "{}" matrix is singular. Fix its positive definiteness.'''.format(mode))
        else:
            d = len(mu)
            theta = np.zeros((d,size))
            C = np.zeros((d,d))
            q = np.zeros(d)
            for i in range(d):
                m1 = np.minimum(i+b+1,d)
                q[i:m1] = A[i:m1,i]
                m = np.maximum(0,i-b)
                for k in range(i-m):
                    k = k + m
                    m2 = np.minimum(k+b+1,d)
                    q[i:m2] = q[i:m2] - C[i:m2,k] * C[i,k]
                C[i:m1,i] = q[i:m1]/sqrt(q[i])
            z = np.random.normal(loc=0,scale=1,size=(np.size(A,0),size))
            for i in range(d):
                j = d - i - 1
                m1 = np.minimum(j+b+1,d)
                theta[j,:] = (z[j,:] - np.dot(C[j:m1,j],theta[j:m1,:]))/C[j,j]
                
            return [np.reshape(mu,(d,1)) + theta,C]
        
    elif mode == "covariance":
        # Check if the matrix is definite positive
        if np.all(np.linalg.eigvals(A) > 0) == False:
            raise ValueError('''The "{}" matrix is singular. Fix its positive definiteness.'''.format(mode))
        else:
            d = len(mu)
            theta = np.zeros((d,size))
            C = np.zeros((d,d))
            q = np.zeros(d)
            for i in range(d):
                m1 = np.minimum(i+b+1,d)
                q[i:m1] = A[i:m1,i]
                m = np.maximum(0,i-b)
                for k in range(i-m):
                    k = k + m
                    m2 = np.minimum(k+b+1,d)
                    q[i:m2] = q[i:m2] - C[i:m2,k] * C[i,k]
                C[i:m1,i] = q[i:m1]/sqrt(q[i])
            z = np.random.normal(loc=0,scale=1,size=(np.size(A,0),size))
            for i in range(d):
                m1 = np.maximum(0,i-b)
                m2 = i + 1
                theta[i,:] = np.dot(C[i,m1:m2],z[m1:m2,:])
                
            return [np.reshape(mu,(d,1)) + theta,C]

    else:
        str_list = ['Invalid mode, choose among:',
                    '- "precision" (default)',
                    '- "covariance"',
                    'Given "{}"'.format(mode)]
        raise ValueError('\n'.join(str_list))


# Multivariate Gaussian sampling with block circulant precision or 
# covariance matrix
def sampler_circulant(mu,a,M,N,mode="precision",seed=None,size=1):
    r"""
    Algorithm dedicated to sample from a multivariate real-valued Gaussian 
    distribution :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{A})` or 
    :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{A}^{-1})` when 
    :math:`\mathbf{A}` is a block-circulant matrix with circulant blocks.
    
    Parameters
    ----------
    mu : 1-D array_like, of length d
        Mean of the d-dimensional Gaussian distribution.
    a : 2-D array_like, of shape (N, M)
        Vector built by stacking the first columns associated to the :math:`M`
        blocks of size :math:`N` of the matrix :math:`\mathbf{A}`.
    M : int
        Number of different blocks.
    N : int
        Dimension of each block.
    mode : string, optional
        Indicates if A refers to the precision or covariance matrix of the
        Gaussian distribution.
    seed : int, optional
           Random seed to reproduce experimental results.
    size : int, optional
        Given a size of for instance T, T independent and identically 
        distributed (i.i.d.) samples are returned.
        
    Returns
    -------
    theta : ndarray, of shape (d,size)
        The drawn samples, of shape (d,size), if that was provided. If not, 
        the shape is (d,1).
        
    Raises
    ------
    ValueError
        If mode is not included in ['covariance','precision'].
        
    Examples
    --------
    >>> d = 2
    >>> mu = np.zeros(d)
    >>> a = np.matrix([1,0]).T
    >>> M = 1
    >>> N = 2
    >>> mode = "covariance"
    >>> size = 1
    >>> theta = sampler_circulant(mu,a,M,N,mode=mode,seed=2022,size=size)
    """ 

    # Set random seed
    np.random.seed(seed)
    
    M = M
    N = N
    mu = np.reshape(mu,(M*N,1))
    if np.size(a,1) == 1:
        Lamb = np.fft.fft(a,axis=0)
    else:
        Lamb = np.fft.fft2(a,axis=0)
        Lamb = np.reshape(Lamb,(M*N,1))

    if mode == "precision":
        # Check if the matrix is definite positive
        if np.all(np.abs(Lamb) > 0) == False:
            raise ValueError('''The "{}" matrix is singular. Fix its positive definiteness.'''.format(mode))
        else:
            z = np.random.normal(loc=0,scale=1,size=(M*N,size))
            return np.fft.ifft(np.fft.fft(mu,axis=0) \
                            + np.fft.fft(z,axis=0) * Lamb**(-1/2),axis=0).real
        
    elif mode == "covariance":
        if np.all(np.abs(Lamb) > 0) == False:
            raise ValueError('''The "{}" matrix is singular. Fix its positive definiteness.'''.format(mode))
        else:
            z = np.random.normal(loc=0,scale=1,size=(M*N,size))
            return np.fft.ifft(np.fft.fft(mu,axis=0) \
                            + np.fft.fft(z,axis=0) * Lamb**(1/2),axis=0).real
        
    else:
        str_list = ['Invalid matrix input, choose among:',
                    '- "precision" (default)',
                    '- "covariance"',
                    'Given "{}"'.format(mode)]
        raise ValueError('\n'.join(str_list))
 
#####################
# General instances #
#####################    
        
# Multivariate Gaussian sampling with factorization 
def sampler_factorization(mu,A,mode="precision",method="Cholesky",seed=None,size=1):
    r"""
    Algorithm dedicated to sample from a multivariate real-valued Gaussian 
    distribution :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{A})` or 
    :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{A}^{-1})` based on matrix
    factorization (e.g., Cholesky or square root).
    
    Parameters
    ----------
    mu : 1-D array_like, of length d
        Mean of the d-dimensional Gaussian distribution.
    A : 2-D array_like, of shape (d, d)
        Covariance or precision matrix of the distribution. It must be 
        symmetric and positive-definite for proper sampling.
    mode : string, optional
        Indicates if A refers to the precision or covariance matrix of the
        Gaussian distribution.
    method : string, optional
        Factorization method. Choose either 'Cholesky' or 'square-root'.
    seed : int, optional
           Random seed to reproduce experimental results.    
    size : int, optional
        Given a size of for instance T, T independent and identically 
        distributed (i.i.d.) samples are returned.
        
    Returns
    -------
    theta : ndarray, of shape (d,size)
        The drawn samples, of shape (d,size), if that was provided. If not, 
        the shape is (d,1).
        
    Raises
    ------
    ValueError
        If A is not positive definite and symmetric.
        If mode is not included in ['covariance','precision'].
        If method is not included in ['Cholesky','square-root'].
        
    Examples
    --------
    >>> d = 2
    >>> mu = np.zeros(d)
    >>> A = np.eye(d)
    >>> mode = "covariance"
    >>> method = "Cholesky"
    >>> size = 1
    >>> theta = sampler_factorization(mu,A,mode=mode,method=method,seed=2022,size=size)
    """ 

    # Set random seed
    np.random.seed(seed)
    
    d = len(mu)
        
    if mode == "precision":
        # Check if the matrix is definite positive
        if np.all(np.linalg.eigvals(A) > 0) == False:
            raise ValueError('''The "{}" matrix is singular. Fix its positive definiteness.'''.format(mode))
        else:
            if method == "Cholesky":
                C = np.linalg.cholesky(A)
                z = np.random.normal(loc=0,scale=1,size=(np.size(A,0),size))
                return np.reshape(mu,(d,1)) \
                       + np.reshape(solve_triangular(C.T,z,lower=False),(d,size))
            elif method == "square-root":
                B = sqrtm(A)
                z = np.random.normal(loc=0,scale=1,size=(np.size(A,0),size))
                return np.reshape(mu,(d,1)) \
                       + np.reshape(np.linalg.solve(B,z),(d,size))
            else:
                str_list = ['Invalid method input, choose among:',
                    '- "Cholesky" (default)',
                    '- "square-root"',
                    'Given "{}"'.format(mode)]
                raise ValueError('\n'.join(str_list))
                
        
    elif mode == "covariance":
         # Check if the matrix is definite positive
        if np.all(np.linalg.eigvals(A) > 0) == False:
            raise ValueError('''The "{}" matrix is singular. Fix its positive definiteness.'''.format(mode))
        else:
            if method == "Cholesky":
                C = np.linalg.cholesky(A)
                z = np.random.normal(loc=0,scale=1,size=(np.size(A,0),size))
                return np.reshape(mu,(d,1)) + np.reshape(C.dot(z),(d,size))
            elif method == "square-root":
                B = sqrtm(A)
                z = np.random.normal(loc=0,scale=1,size=(np.size(A,0),size))
                return np.reshape(mu,(d,1)) + np.reshape(B.dot(z),(d,size))
            else:
                str_list = ['Invalid method input, choose among:',
                    '- "Cholesky" (default)',
                    '- "square-root"',
                    'Given "{}"'.format(mode)]
                raise ValueError('\n'.join(str_list))
      
        
    else:
        str_list = ['Invalid mode input, choose among:',
                    '- "precision" (default)',
                    '- "covariance"',
                    'Given "{}"'.format(mode)]
        raise ValueError('\n'.join(str_list))
        
# Multivariate Gaussian sampling with square-root approximation 
def sampler_squareRootApprox(mu,A,lam_l,lam_u,tol,K=100,mode="precision",seed=None,
                             size=1,info=False):
    r"""
    Algorithm dedicated to sample from a multivariate real-valued Gaussian 
    distribution :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{A})` or 
    :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{A}^{-1})` based on matrix
    square root approximation using Chebychev polynomials.
    
    Parameters
    ----------
    mu : 1-D array_like, of length d
        Mean of the d-dimensional Gaussian distribution.
    A : function
        Linear operator returning the matrix-vector product 
        :math:`\mathbf{Ax}` where :math:`\mathbf{x}) \in \mathbb{R}^d`.
    lam_l : float
        Lower bound on the eigenvalues of :math:`\mathbf{A}`.
    lam_u : float
        Upper bound on the eigenvalues of :math:`\mathbf{A}`.
    tol : float
        Tolerance threshold used to optimize the polynomial order :math:`K`.
        This threshold stands for the Euclidean distance between the vector 
        computed using order :math:`K` and the one computed using order 
        :math:`L`:math:`\leq`:math:`K`.
    K : int, optional
        Polynomial order of the approximation.
    mode : string, optional
        Indicates if A refers to the precision or covariance matrix of the
        Gaussian distribution.
    seed : int, optional
           Random seed to reproduce experimental results.
    size : int, optional
        Given a size of for instance T, T independent and identically 
        distributed (i.i.d.) samples are returned.
    info : boolean, optional
        If info is True, returns the order :math:`K` used in the polynomial
        approximation.
        
    Returns
    -------
    theta : ndarray, of shape (d,size)
        The drawn samples, of shape (d,size), if that was provided. If not, 
        the shape is (d,1).
        
    Raises
    ------
    ValueError
        If mode is not included in ['covariance','precision'].
        
    Examples
    --------
    >>> d = 2
    >>> mu = np.zeros(d)
    >>> def A(x):
        return np.eye(d).dot(x)
    >>> lam_l = 0
    >>> lam_u = 1
    >>> tol = 1e-4
    >>> mode = "covariance"
    >>> size = 1
    >>> theta = sampler_squareRootApprox(mu,A,lam_l=lam_l,lam_u=lam_u,tol=tol,
    mode=mode,seed=2022,size=size)
    """ 

    # Set random seed
    np.random.seed(seed)
        
    if mode == "precision":
        def fun(x):
            return 1/sqrt(x)
        ch = Chebyshev(lam_l, lam_u, K, fun)
        alpha = 2 / (lam_u - lam_l)
        beta = (lam_u + lam_l) / (lam_u - lam_l)
        theta = np.zeros((len(mu),size))
        L = np.linspace(start=1,stop=K,num=50)
        err = np.zeros((len(L),))
        for l in range(len(L)):
            err[l] = np.sum(np.abs(ch.c[int(L[l])+1:K]))
        idx = np.where(err < sqrt(tol*size) \
                       /np.linalg.norm(np.random.normal(0,1,size)))
        if len(idx) > 0:
            K = int(L[np.min(idx)])
        else:
            K = K
            
        z = np.random.normal(0,1,size=(len(mu),size))
        u1 = alpha * A(z) - beta * z
        u0 = z
        u = 0.5 * ch.c[0] * u0 + ch.c[1] * u1
        k = 2
        while(k <= K - 1):
            ubis = 2 * (alpha * A(u1) - beta * u1) - u0
            u = u + ch.c[k] * ubis
            u0 = u1
            u1 = ubis
            k = k + 1
        theta = np.reshape(mu,(len(mu),1)) + u
        
        if info == True:
            return (theta,K)
        else:
            return theta
                
        
    elif mode == "covariance":
        ch = Chebyshev(lam_l, lam_u, K, sqrt)
        alpha = 2 / (lam_u - lam_l)
        beta = (lam_u + lam_l) / (lam_u - lam_l)
        theta = np.zeros((len(mu),size))
        L = np.linspace(start=1,stop=K,num=50)
        err = np.zeros((len(L),))
        for l in range(len(L)):
            err[l] = np.sum(np.abs(ch.c[int(L[l])+1:K]))
        idx = np.where(err < sqrt(tol*size) \
                       /np.linalg.norm(np.random.normal(0,1,size)))
        if len(idx) > 0:
            K = int(L[np.min(idx)])
        else:
            K = K
            
        z = np.random.normal(0,1,size=(len(mu),size))
        u1 = alpha * A(z) - beta * z
        u0 = z
        u = 0.5 * ch.c[0] * u0 + ch.c[1] * u1
        k = 2
        while(k <= K - 1):
            ubis = 2 * (alpha * A(u1) - beta * u1) - u0
            u = u + ch.c[k] * ubis
            u0 = u1
            u1 = ubis
            k = k + 1
        theta = np.reshape(mu,(len(mu),1)) + u
        
        if info == True:
            return (theta,K)
        else:
            return theta
        
    else:
        str_list = ['Invalid mode input, choose among:',
                    '- "precision" (default)',
                    '- "covariance"',
                    'Given "{}"'.format(mode)]
        raise ValueError('\n'.join(str_list))
        
        
# Multivariate Gaussian sampling with conjugate gradients 
def sampler_CG(mu,A,K,init,tol=1e-4,mode="precision",seed=None,size=1,info=False):
    r"""
    Algorithm dedicated to sample from a multivariate real-valued Gaussian 
    distribution :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{A})` or 
    :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{A}^{-1})` based on the
    conjugate gradient algorithm.
    
    Parameters
    ----------
    mu : 1-D array_like, of length d
        Mean of the d-dimensional Gaussian distribution.
    A : function
        Linear operator returning the matrix-vector product 
        :math:`\mathbf{Ax}` where :math:`\mathbf{x} \in \mathbb{R}^d`.
    K : int, optional
        Number of conjugate gradient iterations.
    init : 1-D array_like, of length d
        Vector used to initialize the CG sampler.
    tol : float, optional
        Tolerance threshold used to stop the conjugate gradient sampler.
    mode : string, optional
        Indicates if A refers to the precision or covariance matrix of the
        Gaussian distribution.
    seed : int, optional
           Random seed to reproduce experimental results.
    size : int, optional
        Given a size of for instance T, T independent and identically 
        distributed (i.i.d.) samples are returned.
    info : boolean, optional
        If info is True, returns the number of iterations :math:`K`.
        
    Returns
    -------
    theta : ndarray, of shape (d,size)
        The drawn samples, of shape (d,size), if that was provided. If not, 
        the shape is (d,1).
        
    Raises
    ------
    ValueError
        If mode is not included in ['covariance','precision'].
        
    Examples
    --------
    >>> d = 2
    >>> mu = np.zeros(d)
    >>> def A(x):
        return np.eye(d).dot(x)
    >>> K = 2
    >>> init = mu
    >>> theta = sampler_CG(mu,A,K,init)
    """ 

    # Set random seed
    np.random.seed(seed)
    
    d = len(mu)
    init = np.reshape(init,(d,1))
    mu = np.reshape(mu,(d,1))
    loss_conj = 0
        
    if mode == "precision":
        theta = np.zeros((d,size))
        # Initialization
        k = 1
        r_old = np.random.normal(0,1,size=(d,size)) - A(init)
        #rd = np.random.randint(0, 2, (d, size))
        #rd[rd == 0] = -1
        #r_old = rd - B(init)
        p_old = r_old
        d_old = (p_old * A(p_old)).sum(axis=0)
        y = init
        r_new = np.ones((d,size))
        loss_conj = 0
        while (col_vector_norms(r_new,2) >= tol).any() and k <= K:
            gam = (r_old * r_old).sum(axis=0) / d_old
            z = np.random.normal(0,1,size=size)
            y = y + z / np.sqrt(d_old) * p_old
            r_new = r_old - gam * A(p_old)
            beta = - (r_new * r_new).sum(axis=0) / (r_old * r_old).sum(axis=0)
            p_new = r_new - beta * p_old
            if (np.abs((p_new * A(p_old)).sum(axis=0)) >= 1e-4).any() and loss_conj == 0 and info==True:
                print('Loss of conjugacy happened at iteration k = %i.'%k)
                loss_conj = 1
                k_loss = k
            d_new = (p_new * A(p_new)).sum(axis=0)
            r_old = r_new
            p_old = p_new
            d_old = d_new
            k = k + 1
        theta = mu + y
        
        if info == True and loss_conj == 1:
            return (theta,k-1,loss_conj,k_loss)
        elif info == True and loss_conj == 0:
            return (theta,k-1,loss_conj)
        else:
            return theta
    
    elif mode == "covariance":
        theta = np.zeros((d,size))
        # Initialization
        k = 1
        r_old = np.random.normal(0,1,size=(d,size)) - A(init)
        #rd = np.random.randint(0, 2, (d, size))
        #rd[rd == 0] = -1
        #r_old = rd - A(init)
        p_old = r_old
        d_old = (p_old * A(p_old)).sum(axis=0)
        y = init
        r_new = np.ones((d,size))
        loss_conj = 0
        while (col_vector_norms(r_new,2) >= tol).any() and k <= K:
            gam = (r_old * r_old).sum(axis=0) / d_old
            z = np.random.normal(0,1,size=size)
            y = y + z / np.sqrt(d_old) * A(p_old)
            r_new = r_old - gam * A(p_old)
            beta = - (r_new * r_new).sum(axis=0) / (r_old * r_old).sum(axis=0)
            p_new = r_new - beta * p_old
            if (np.abs((p_new * A(p_old)).sum(axis=0)) >= 1e-4).any() and loss_conj == 0 and info==True:
                print('Loss of conjugacy happened at iteration k = %i.'%k)
                loss_conj = 1
                k_loss = k
            d_new = (p_new * A(p_new)).sum(axis=0)
            r_old = r_new
            p_old = p_new
            d_old = d_new
            k = k + 1
        theta = mu + y
        
        if info == True and loss_conj == 1:
            return (theta,k-1,loss_conj,k_loss)
        elif info == True and loss_conj == 0:
            return (theta,k-1,loss_conj)
        else:
            return theta
    
    else:
        str_list = ['Invalid mode input, choose among:',
                    '- "precision" (default)',
                    '- "covariance"',
                    'Given "{}"'.format(mode)]
        raise ValueError('\n'.join(str_list))
    
# Multivariate Gaussian sampling with (truncated) - perturbation-optimization
class sampler_PO:
    r"""
    Algorithm dedicated to sample from a multivariate real-valued Gaussian 
    distribution :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{Q}^{-1})` where
    :math:`\mathbf{Q}` is a symmetric and positive definite precision matrix.
    We assume here that :math:`\mathbf{Q} = 
    \mathbf{G}_1^T\mathbf{\Lambda}_1^{-1}\mathbf{G}_1 + 
    \mathbf{G}_2^T\mathbf{\Lambda}_2^{-1}\mathbf{G}_2`. The mean vector is
    assumed to have the form :math:`\boldsymbol{\mu} = 
    \mathbf{G}_1^T\mathbf{\Lambda}_1^{-1}\boldsymbol{\mu}_1 + 
    \mathbf{G}_2^T\mathbf{\Lambda}_2^{-1}\boldsymbol{\mu}_2`. Sampling from the
    corresponding multivariate Gaussian distribution is done with the 
    perturbation-optimization sampler.
    """

    def __init__(self,mu1,mu2,K,init,tol=1e-4,seed=None,size=1):
        r"""
    
        Parameters
        ----------
        mu1 : 1-D array_like, of length d
        mu2 : 1-D array_like, of length d
        K : int, optional
            Number of conjugate gradient iterations to solve the linear system
            :math:`\mathbf{Q}\boldsymbol{\theta} = \boldsymbol{\eta}`.
        init : 1-D array_like, of length d
            Vector used to initialize the CG algorithm.
        tol : float, optional
            Tolerance threshold used to stop the conjugate gradient algorithm.
        seed : int, optional
               Random seed to reproduce experimental results.
        size : int, optional
            Given a size of for instance T, T independent and identically 
            distributed (i.i.d.) samples are returned.
            """
            
        self.mu1 = mu1
        self.mu2 = mu2
        self.K = K
        self.init = init
        self.tol = tol
        self.seed = seed
        self.size = size
    
    def circu_diag_band(self,Lamb1,g,M,N,Q2,b2):
        r"""
        We assume here that :math:`\mathbf{G}_1` is a circulant matrix, 
        :math:`\mathbf{\Lambda}_1` is diagonal, :math:`\mathbf{G}_2` is the
        identity matrix and :math:`\mathbf{Q}_2 = \mathbf{\Lambda}_2^{-1}` is 
        a band matrix.
        
        Parameters
        ----------
        Lamb1 : 1-D array_like, of length d
            Diagonal elements of :math:`\mathbf{\Lambda}_1`.
        g : 2-D array_like, of shape (N, M)
            Vector built by stacking the first columns associated to the 
            :math:`M` blocks of size :math:`N` of the matrix 
            :math:`\mathbf{G}_1`.
        M : int
            Number of different blocks in :math:`\mathbf{G}_1`.
        N : int
            Dimension of each block in :math:`\mathbf{G}_1`.
        Q2 : 2-D array_like, of shape (d, d)
            Precision matrix :math:`\mathbf{Q}_2`.
        b2 : int
            Bandwidth of :math:`\mathbf{Q}_2`.
            
        Returns
        -------
        theta : ndarray, of shape (d,size)
            The drawn samples, of shape (d,size), if that was provided. If not, 
            the shape is (d,1).
            
        Examples
        --------
        >>> d = 15
        >>> mu1 = np.zeros(d)
        >>> mu2 = np.zeros(d)
        >>> K = 15
        >>> init = np.zeros(d)
        >>> Lamb1 = np.random.normal(2,0.1,d)
        >>> g =  np.reshape(np.random.normal(2,0.1,d),(d,1))
        >>> M = 1
        >>> N = d
        >>> Q2 = np.diag(np.random.normal(2,0.1,d))
        >>> b2 = 0
        >>> size = 10000
        >>> S = sampler_PO(mu1,mu2,K,init,size=10000)
        >>> theta = S.circu_diag_band(Lamb1,g,M,N,Q21,b2)
        """ 

        # Set random seed
        np.random.seed(self.seed)

        d = len(Lamb1)
        eta1 = np.reshape(self.mu1,(d,1)) \
               + np.random.normal(0,1,size=(d,self.size)) \
               * np.sqrt(Lamb1)[:, np.newaxis]
        [eta2,C] = sampler_band(self.mu2,Q2,b2,mode="precision",size=self.size)
        
        def Q2(x):
            CTx = np.zeros((M*N,self.size))
            Q2x = CTx 
            for i in range(M*N):
                m1 = i
                m2 = b2 + i + 1
                CTx[i,:] = np.dot(C.T[i,m1:m2],x[m1:m2,:])
            for i in range(M*N):
                m1 = np.maximum(0,i-b2)
                m2 = i + 1
                Q2x[i,:] = np.dot(C[i,m1:m2],CTx[m1:m2,:])
            return Q2x
        
        if np.size(g,1) == 1:
            LambG1 = np.fft.fft(g,axis=0)
            LambG1 = np.reshape(LambG1,(M*N,1))
        else:
            LambG1 = np.fft.fft2(g,axis=0)
            LambG1 = np.reshape(LambG1,(M*N,1))
        
        eta = np.fft.ifft(np.fft.fft(eta1 / Lamb1[:, np.newaxis],axis=0) \
                  * LambG1.conj(),axis=0).real + Q2(eta2)
    
        def Q(x):
            return np.fft.ifft(np.fft.fft(np.fft.ifft(np.fft.fft(x,axis=0) \
                   * LambG1,axis=0).real / Lamb1[:,np.newaxis],axis=0) \
                   * LambG1.conj(),axis=0).real + Q2(x)
        
        return CG(Q,eta,x=None,K=None,tol=None,info=False)
        
        
        
