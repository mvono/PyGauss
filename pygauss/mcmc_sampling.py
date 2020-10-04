#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Implementation of Markov chain Monte Carlo (MCMC) approaches to sample 
from multivariate Gaussian distributions.

.. seealso::
    
    `Documentation on ReadTheDocs <https://pygauss-gaussian-sampling.readthedocs.io/en/latest/mcmc_sampling/index.html>`_
"""

from direct_sampling import sampler_squareRootApprox, sampler_band
from utils import diagonal_form, triangular_inversion
import numpy as np
from scipy.linalg import solve_triangular, solve_banded

#####################
# General instances #
#####################    
        
# MCMC sampling based on matrix splitting 
class sampler_MS:
    r"""
    Algorithm dedicated to sample from a multivariate real-valued Gaussian 
    distribution :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{Q}^{-1})` where
    :math:`\mathbf{Q}` is a symmetric and positive definite precision matrix.
    We assume here that the matrix splitting scheme :math:`\mathbf{Q} = 
    \mathbf{M} - \mathbf{N}` holds.
    """

    def __init__(self,mu,Q,ini,b,band=True,size=1):
        r"""
    
        Parameters
        ----------
        mu : 1-D array_like, of length d
        size : int, optional
            Given a size of for instance T, T independent and identically 
            distributed (i.i.d.) samples are returned.
        Q : 2-D array_like, of shape (d,d)
            Precision matrix.
            
            """
            
        self.mu = mu
        self.size = size
        self.Q = Q
        self.ini = ini
        self.b = b
        self.band = band

    def exact_MS(self,method="Gauss-Seidel"):
        r"""
        The samplers considered here are exact.
        
        Parameters
        ----------
        method : string, optional
            Matrix splitting approach to choose within 
            ['Gauss-Seidel','Richardson','Jacobi','SOR'].
            
        Returns
        -------
        theta : ndarray, of shape (d,size)
            The drawn samples, of shape (d,size), if that was provided. If not, 
            the shape is (d,1).
            
        Examples
        --------
        >>> import mcmc_sampling as mcmc
        >>> d = 10
        >>> mu = np.zeros(d)
        >>> Q = np.eye(d)
        >>> S = mcmc.sampler_MS(mu,Q,size=1)
        >>> theta = S.exact_MS(method="Gauss-Seidel")
        """ 
        
        d = len(self.mu)
        theta = np.zeros((d,self.size))
        theta[:,0] = self.ini
        
        if method == "Gauss-Seidel":
                
            # Matrix splitting Q = M - N
            M = np.tril(self.Q)
            D = np.diag(self.Q)
            def N(x):
                mat = - (np.triu(self.Q) - np.diag(D))
                return mat.dot(x)
                
            # Gibbs sampling
            for t in range(self.size-1):
                z = np.random.normal(0,1,size=d) * np.sqrt(D)
                Qtheta = N(theta[:,t]) + z
                theta[:,t+1] = solve_triangular(M,Qtheta,lower=True)
            
        elif method == "Richardson":
                
            # Matrix splitting Q = M - N
            omega = 2/(np.max(np.abs(np.linalg.eigvals(self.Q))) + np.min(np.abs(np.linalg.eigvals(self.Q)))) 
            M = np.ones(d) / omega
            N = np.diag(M) - self.Q
            cov = np.diag(2 * M) - self.Q
            def A(x):
                mat = np.diag(2 * M) - self.Q
                return mat.dot(x)
            lam_u = np.max(np.sum(np.abs(A(np.eye(d))),0))
                
            # Gibbs sampling
            for t in range(self.size-1):
                if self.band == True and t == 0:
                    [z,C] = sampler_band(np.zeros(d),cov,self.b,mode="covariance",size=1)
                elif self.band == True and t > 0:
                    z = C.dot(np.random.normal(0,1,size=d))
                else:
                    z = sampler_squareRootApprox(np.zeros(d),A,lam_l=0,
                                             lam_u=lam_u,tol=1e-2,
                                             K=d,mode='covariance')
                Qtheta = N.dot(theta[:,t]) + np.reshape(z,(d,))
                theta[:,t+1] = Qtheta / M
                
        elif method == "Jacobi":
            
            # Check if Q is strictly diagonally dominant
            D = np.diag(np.abs(self.Q)) 
            S = np.sum(np.abs(self.Q), axis=1) - D 
            if np.all(D <= S):
                raise ValueError('''The precision matrix Q is not strictly diagonally dominant. The Gibbs sampler does not converge.''')

            # Matrix splitting Q = M - N
            M = np.diag(self.Q)
            N = np.diag(M) - self.Q
            cov = np.diag(2 * M) - self.Q
            def A(x):
                mat = np.diag(2 * M) - self.Q
                return mat.dot(x)
            lam_u = np.max(np.sum(np.abs(A(np.eye(d))),0))
                
            # Gibbs sampling
            for t in range(self.size-1):
                if self.band == True and t == 0:
                    [z,C] = sampler_band(np.zeros(d),cov,self.b,mode="covariance",size=1)
                elif self.band == True and t > 0:
                    z = C.dot(np.random.normal(0,1,size=d))
                else:
                    z = sampler_squareRootApprox(np.zeros(d),A,lam_l=0,
                                             lam_u=lam_u,tol=1e-2,
                                             K=d,mode='covariance')
                Qtheta = N.dot(theta[:,t]) + np.reshape(z,(d,))
                theta[:,t+1] = Qtheta / M
                
        elif method == "SOR":
            
            # Check if Q is strictly diagonally dominant
            D = np.diag(np.abs(self.Q)) 
            S = np.sum(np.abs(self.Q), axis=1) - D 
            if np.all(D <= S):
                omega = 1.7
                print('''The precision matrix Q is not strictly diagonally dominant. A default value has been set for the relaxation parameter omega = %f.'''%omega)
            else:
                Dinv = 1 / np.diag(self.Q)
                J = np.eye(d) - self.Q * Dinv[:,None]
                rho = np.max(np.abs(np.linalg.eigvals(J)))
                omega = 2 / (1 + np.sqrt(1 - rho**2))
                
            # Matrix splitting Q = M - N
            M = np.tril(self.Q) \
                + (1-omega)/omega * np.diag(self.Q) * np.eye(d)
            D = (2-omega)/omega * np.diag(self.Q)
            def N(x):
                mat = - (np.triu(self.Q) - np.diag(self.Q) * np.eye(d)) \
                      + (1-omega)/omega * np.diag(self.Q) * np.eye(d)
                return mat.dot(x)
                
            # Gibbs sampling
            for t in range(self.size-1):
                z = np.random.normal(0,1,size=d) * np.sqrt(D)
                Qtheta = N(theta[:,t]) + z
                theta[:,t+1] = solve_triangular(M,Qtheta,lower=True)
                
        elif method == "SSOR":
            
            # Check if Q is strictly diagonally dominant
            D = np.diag(np.abs(self.Q)) 
            S = np.sum(np.abs(self.Q), axis=1) - D 
            if np.all(D <= S):
                omega = 1.5
                print('''The precision matrix Q is not strictly diagonally dominant. A default value has been set for the relaxation parameter omega.''')
            else:
                Dinv = 1 / np.diag(self.Q)
                J = np.eye(d) - self.Q * Dinv[:,None]
                rho = np.max(np.abs(np.linalg.eigvals(J)))
                omega = 2 / (1 + np.sqrt(1 - rho**2))
                
            # Matrix splitting Q = M - N
            M = np.tril(self.Q) \
                + (1-omega)/omega * np.diag(self.Q) * np.eye(d)
            D = (2-omega)/omega * np.diag(self.Q)
            def N(x):
                mat = - (np.triu(self.Q) - np.diag(self.Q) * np.eye(d)) \
                      + (1-omega)/omega * np.diag(self.Q) * np.eye(d)
                return mat.dot(x)
            def NT(x):
                mat = - (np.triu(self.Q) - np.diag(self.Q) * np.eye(d)) \
                      + (1-omega)/omega * np.diag(self.Q) * np.eye(d)
                return mat.T.dot(x)
                
            # Gibbs sampling
            for t in range(self.size-1):
                z = np.random.normal(0,1,size=d) * np.sqrt(D)
                Qtheta = N(theta[:,t]) + z
                theta_bis = solve_triangular(M,Qtheta,lower=True)
                z = np.random.normal(0,1,size=d) * np.sqrt(D)
                Qtheta = NT(theta_bis) + z
                theta[:,t+1] = solve_triangular(M.T,Qtheta,lower=False)
                
        
        return np.reshape(self.mu,(d,1)) + theta  
    
            
    def approx_MS(self,method="Clone-MCMC",omega=1):
        r"""
        The samplers considered here are approximate.
        
        Parameters
        ----------
        method : string, optional
            Matrix splitting approach to choose within 
            ['Gauss-Seidel','Richardson','Jacobi','SOR'].
        omega : float, optional
            Tuning parameter appearing in some approximate matrix splitting 
            Gibbs samplers.
            
        Returns
        -------
        theta : ndarray, of shape (d,size)
            The drawn samples, of shape (d,size), if that was provided. If not, 
            the shape is (d,1).
            
        Examples
        --------
        >>> import mcmc_sampling as mcmc
        >>> d = 10
        >>> mu = np.zeros(d)
        >>> Q = np.eye(d)
        >>> S = mcmc.sampler_MS(mu,Q,size=1)
        >>> theta = S.approx_MS(method="Gauss-Seidel",omega=1)
        """ 
        
        d = len(self.mu)
        theta = np.zeros((d,self.size))
        
        if method == "Clone-MCMC":
            
            # Check if Q is strictly diagonally dominant
            D = np.diag(np.abs(self.Q)) 
            S = np.sum(np.abs(self.Q), axis=1) - D 
            if np.all(D <= S):
                raise ValueError('''The precision matrix Q is not strictly diagonally dominant. The Gibbs sampler does not converge.''')

                
            # Matrix splitting Q = M - N
            M = np.diag(self.Q) + 2 * omega
            Mbis = 2 * M
            def N(x):
                mat = 2 * omega * np.eye(d) \
                      - np.triu(self.Q) - np.tril(self.Q) \
                      + 2 * np.diag(np.diag(self.Q))
                return mat.dot(x)
                
            # Gibbs sampling
            for t in range(self.size-1):
                z = np.random.normal(0,1,size=d) * np.sqrt(Mbis)
                Qtheta = N(theta[:,t]) + z
                theta[:,t+1] = Qtheta / M
                
        if method == "Hogwild":
            
            # Check if Q is strictly diagonally dominant
            D = np.diag(np.abs(self.Q)) 
            S = np.sum(np.abs(self.Q), axis=1) - D 
            if np.all(D <= S):
                raise ValueError('''The precision matrix Q is not strictly diagonally dominant. The Gibbs sampler does not converge.''')

                
            # Matrix splitting Q = M - N
            M = np.diag(self.Q)
            def N(x):
                mat = - np.triu(self.Q) - np.tril(self.Q) \
                      + 2 * np.diag(np.diag(self.Q))
                return mat.dot(x)
                
            # Gibbs sampling
            for t in range(self.size-1):
                z = np.random.normal(0,1,size=d) * np.sqrt(M)
                Qtheta = N(theta[:,t]) + z
                theta[:,t+1] = Qtheta / M

        return np.reshape(self.mu,(d,1)) + theta  
        
        
# MCMC sampling based on data augmentation
class sampler_DA:
    r"""
    Algorithm dedicated to sample from a multivariate real-valued Gaussian 
    distribution :math:`\mathcal{N}(\boldsymbol{\mu},\mathbf{Q}^{-1})` where
    :math:`\mathbf{Q}` is a symmetric and positive definite precision matrix.
    We assume here that :math:`\mathbf{Q} = 
    \mathbf{G}_1^T\mathbf{\Lambda}_1^{-1}\mathbf{G}_1 + 
    \mathbf{G}_2^T\mathbf{\Lambda}_2^{-1}\mathbf{G}_2`. Sampling from the
    corresponding multivariate Gaussian distribution is done with an MCMC
    algorithm based on a data augmentation scheme.
    """

    def __init__(self,mu,size=1):
        r"""
    
        Parameters
        ----------
        mu : 1-D array_like, of length d
        size : int, optional
            Given a size of for instance T, T independent and identically 
            distributed (i.i.d.) samples are returned.       
        """
            
        self.mu = mu
        self.size = size
    
    def exact_DA_circu_diag_band(self,Lamb1,g,M,N,Q2,b2,method="GEDA"):
        r"""
        The samplers considered here are exact. We further assume here 
        that :math:`\mathbf{G}_1` is a circulant matrix, 
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
        method : string, optional
            Data augmentation approach to choose within ['EDA','GEDA'].
            
        Returns
        -------
        theta : ndarray, of shape (d,size)
            The drawn samples, of shape (d,size), if that was provided. If not, 
            the shape is (d,1).
            
        Examples
        --------
        >>> import mcmc_sampling as mcmc
        >>> d = 15
        >>> Lamb1 = np.random.normal(2,0.1,d)
        >>> g =  np.reshape(np.random.normal(2,0.1,d),(d,1))
        >>> M = 1
        >>> N = d
        >>> Q2 = np.diag(np.random.normal(2,0.1,d))
        >>> b2 = 0
        >>> S = mcmc.sampler_DA(mu,size=1)
        >>> theta = S.exact_DA_circu_diag_band(Lamb1,g,M,N,
                                               Q2,b2,method="EDA")
        """ 
        
        # Pre-computing
        if np.size(g,1) == 1:
            LambG1 = np.fft.fft(g,axis=0)
            LambG1 = np.reshape(LambG1,(M*N,1))
        else:
            LambG1 = np.fft.fft2(g,axis=0)
            LambG1 = np.reshape(LambG1,(M*N,1))
        
        def Q1(x):
            return np.fft.ifft(np.fft.fft(np.fft.ifft(np.fft.fft(x,axis=0) \
                   * LambG1,axis=0).real / np.reshape(Lamb1,(d,1)),axis=0) \
                   * LambG1.conj(),axis=0).real
        
        omega = (0.5 / np.max(np.abs(LambG1))**2) * np.min(Lamb1)**2
            
        d = len(self.mu)
        theta = np.zeros((d,self.size))
        
        if method == "EDA":
            
            for t in range(self.size-1):
            
                # Sample the auxiliary variable u1
                mu_u1 = np.reshape(np.reshape(theta[:,t],(d,1)) / omega \
                                   - Q1(np.reshape(theta[:,t],(d,1))),(d,))
                def A(x):
                    return x / omega - Q1(x)
                lam_u = 1/omega
                u1 = sampler_squareRootApprox(mu_u1,A,lam_l=0,
                                             lam_u=lam_u,tol=1e-2,
                                             K=d,mode='covariance')
                u1 = np.reshape(u1,(d,))
                
                # Sample the variable of interest theta
                Q_theta = np.eye(d) / omega + Q2
                z = sampler_band(np.zeros(d),Q_theta,b2,mode="precision",
                                        size=1)[0]
                C = sampler_band(np.zeros(d),Q2,b2,mode="precision",
                                        size=1)[1]
                def Q2_fun(x):
                    CTx = np.zeros(M*N)
                    Q2x = CTx 
                    for i in range(M*N):
                        m1 = i
                        m2 = b2 + i + 1
                        CTx[i] = np.dot(C.T[i,m1:m2],x[m1:m2])
                    for i in range(M*N):
                        m1 = np.maximum(0,i-b2)
                        m2 = i + 1
                        Q2x[i] = np.dot(C[i,m1:m2],CTx[m1:m2])
                    return Q2x
                mu_theta = u1 + np.reshape(Q1(np.reshape(self.mu,(d,1))),(d,)) + Q2_fun(self.mu)
                ab = diagonal_form(Q_theta,lower=b2,upper=b2)
                theta[:,t+1] = solve_banded((b2, b2), ab, mu_theta) \
                               + np.reshape(z,(d,))
                               
        if method == "GEDA":
            
            u1 = np.zeros(d)
            
            for t in range(self.size-1):
            
                # Sample the auxiliary variable u2
                mu_u2 = np.fft.ifft(np.fft.fft(u1,axis=0) \
                                    * np.reshape(LambG1,(d,)),axis=0).real
                u2 = mu_u2 + np.random.normal(0,1,size=d) * np.sqrt(1/Lamb1)
                
                # Sample the auxiliary variable u1
                mu_u1 = theta[:,t] - omega * \
                        np.reshape(Q1(np.reshape(theta[:,t],(d,1))),(d,)) \
                        + omega * np.fft.ifft(np.reshape(LambG1.conj(),(d,)) \
                        * np.fft.fft(1/Lamb1 * u2,axis=0),axis=0).real
                u1 = mu_u1 + np.random.normal(0,1,size=d) * np.sqrt(omega)
                
                # Sample the variable of interest theta
                Q_theta = np.eye(d) / omega + Q2
                z = sampler_band(np.zeros(d),Q_theta,b2,mode="precision",
                                        size=1)[0]
                C = sampler_band(np.zeros(d),Q2,b2,mode="precision",
                                        size=1)[1]
                def Q2_fun(x):
                    CTx = np.zeros(M*N)
                    Q2x = CTx 
                    for i in range(M*N):
                        m1 = i
                        m2 = b2 + i + 1
                        CTx[i] = np.dot(C.T[i,m1:m2],x[m1:m2])
                    for i in range(M*N):
                        m1 = np.maximum(0,i-b2)
                        m2 = i + 1
                        Q2x[i] = np.dot(C[i,m1:m2],CTx[m1:m2])
                    return Q2x
                mu_theta = u1 / omega \
                           - np.reshape(Q1(np.reshape(u1,(d,1))),(d,)) \
                           + np.reshape(Q1(np.reshape(self.mu,(d,1))),(d,)) \
                           + Q2_fun(self.mu)
                ab = diagonal_form(Q_theta,lower=b2,upper=b2)
                theta[:,t+1] = solve_banded((b2, b2), ab, mu_theta) \
                               + np.reshape(z,(d,))
                
        return theta  

    def exact_DA_circu_diag_circu(self,Lamb1,LambG1,LambG2,A,method="GEDA"):
        r"""
        The samplers considered here are exact. We further assume here 
        that :math:`\mathbf{G}_1` is a circulant matrix, 
        :math:`\mathbf{\Lambda}_1` is diagonal, :math:`\mathbf{\Lambda}_2` is the identity matrix
        and :math:`\mathbf{G}_2` is a circulant matrix.
        
        Parameters
        ----------
        Lamb1 : 1-D array_like, of length d
            Diagonal elements of :math:`\mathbf{\Lambda}_1`.
        LambG1 : 1-D array_like, of length d
            Diagonal elements of the Fourier counterpart matrix associated to the matrix 
            :math:`\mathbf{G}_1`.
        LambG2 : 1-D array_like, of length d
            Diagonal elements of the Fourier counterpart matrix associated to the matrix 
            :math:`\mathbf{G}_2`.
        A : function
            Linear operator returning the matrix-vector product 
            :math:`\mathbf{Qx}` where :math:`\mathbf{x} \in \mathbb{R}^d`.
        method : string, optional
            Data augmentation approach to choose within ['EDA','GEDA'].
            
        Returns
        -------
        theta : ndarray, of shape (d,size)
            The drawn samples, of shape (d,size), if that was provided. If not, 
            the shape is (d,1).
            
        Examples
        --------
        >>> import mcmc_sampling as mcmc
        >>> d = 15
        >>> Lamb1 = np.random.normal(2,0.1,d)
        >>> g =  np.reshape(np.random.normal(2,0.1,d),(d,1))
        >>> M = 1
        >>> N = d
        >>> Q2 = np.diag(np.random.normal(2,0.1,d))
        >>> b2 = 0
        >>> S = mcmc.sampler_DA(mu,size=1)
        >>> theta = S.exact_DA_circu_diag_band(Lamb1,g,M,N,
                                               Q2,b2,method="EDA")
        """ 

        def Q1(x):
            Fx = np.fft.fft(x,axis=0)
            return np.fft.ifft(np.fft.fft(np.fft.ifft(Fx * LambG1,axis=0).real \
                       * (1/Lamb1),axis=0) * LambG1.conj(),axis=0).real
        
        omega = 0.9 * np.min(Lamb1) / np.max(np.abs(LambG1))**2

        Lamb = 1/omega + np.abs(LambG2)**2
            
        d = len(self.mu)
        theta = np.zeros((d,self.size))
        
        if method == "EDA":
            
            for t in range(self.size-1):
            
                # Sample the auxiliary variable u1
                mu_u1 = np.reshape(np.reshape(theta[:,t],(d,1)) / omega \
                                   - Q1(np.reshape(theta[:,t],(d,1))),(d,))
                def A(x):
                    return x / omega - Q1(x)
                lam_u = 1/omega
                u1 = sampler_squareRootApprox(mu_u1,A,lam_l=0,
                                             lam_u=lam_u,tol=1e-2,
                                             K=d,mode='covariance')
                u1 = np.reshape(u1,(d,))
                
                # Sample the variable of interest theta
                Q_theta = np.eye(d) / omega + Q2
                z = sampler_band(np.zeros(d),Q_theta,b2,mode="precision",
                                        size=1)[0]
                C = sampler_band(np.zeros(d),Q2,b2,mode="precision",
                                        size=1)[1]
                def Q2_fun(x):
                    CTx = np.zeros(M*N)
                    Q2x = CTx 
                    for i in range(M*N):
                        m1 = i
                        m2 = b2 + i + 1
                        CTx[i] = np.dot(C.T[i,m1:m2],x[m1:m2])
                    for i in range(M*N):
                        m1 = np.maximum(0,i-b2)
                        m2 = i + 1
                        Q2x[i] = np.dot(C[i,m1:m2],CTx[m1:m2])
                    return Q2x
                mu_theta = u1 + np.reshape(Q1(np.reshape(self.mu,(d,1))),(d,)) + Q2_fun(self.mu)
                ab = diagonal_form(Q_theta,lower=b2,upper=b2)
                theta[:,t+1] = solve_banded((b2, b2), ab, mu_theta) \
                               + np.reshape(z,(d,))
                               
        if method == "GEDA":
            
            u1 = np.zeros(d)
            
            for t in range(self.size-1):
            
                # Sample the auxiliary variable u2
                mu_u2 = np.fft.ifft(np.fft.fft(u1,axis=0) \
                                    * np.reshape(LambG1,(d,)),axis=0).real
                u2 = mu_u2 + np.random.normal(0,1,size=d) * np.sqrt(1/Lamb1)
                
                # Sample the auxiliary variable u1
                mu_u1 = theta[:,t] - omega * Q1(theta[:,t]) \
                        + omega * np.fft.ifft(np.reshape(LambG1.conj(),(d,)) \
                        * np.fft.fft(1/Lamb1 * u2,axis=0),axis=0).real
                u1 = mu_u1 + np.random.normal(0,1,size=d) * np.sqrt(omega)
                
                # Sample the variable of interest theta
                z = np.random.normal(loc=0,scale=1,size=d)
                mu_theta = np.fft.ifft(np.fft.fft(u1/omega - Q1(u1) + np.reshape(A(np.reshape(self.mu,(d,1))),(d,)),axis=0) * (1/Lamb),axis=0).real
                theta[:,t+1] = mu_theta + np.fft.ifft(np.fft.fft(z,axis=0) * Lamb**(-1/2),axis=0).real
            
        return theta  
    
    
      
        
        