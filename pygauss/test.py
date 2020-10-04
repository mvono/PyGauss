#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:47:54 2019

@author: maximevono
"""
import numpy as np
from . import direct_sampling
from scipy.linalg import circulant
from sklearn.covariance import EmpiricalCovariance

# Singular circulant precision matrix
c = np.array([4,-1,0,0,-1,0,0,0,0,0,-1,0,0,0,-1])
d = len(c)
C = circulant(c)
Q = np.dot(C.T,C)  
theta = sampler_circulant(circu_vec=c,
                  num_block=1,
                  dim_block=d,
                  mean=np.zeros((d,)),
                  matrix="precision",
                  num_samples=1000)

# Fixing the positive definiteness of the precision matrix
gam = 1
c = np.array([4,-1,0,0,-1,0,0,0,0,0,-1,0,0,0,-1])
c = c + gam/len(c)
C = circulant(c)
Q = np.dot(C.T,C) + gam*np.ones(len(c))    
mu = np.zeros((d,))
theta = sampler_circulant(a=np.reshape(Q[:,0],(len(c),1)),
                  M=1,
                  N=len(c),
                  mu=mu,
                  mode="precision",
                  size=10000)
mu_hat = np.mean(theta,axis=1)
Q_hat = EmpiricalCovariance().fit(theta.T).precision_
np.linalg.norm(mu-mu_hat)
np.linalg.norm(Q-Q_hat)/np.linalg.norm(Q)


mu = np.array([0., 1.])
Sigma = np.array([[ 1. , -0.9], [-0.9,  1]])
Q = np.linalg.inv(Sigma)
def matvec_fun(x):
    return Q.dot(x)
lam_l = 0
lam_u = np.max(np.sum(np.abs(Q),0))
tol = 1e-3
[theta,K] = sampler_squareRootApprox(mu,matvec_fun,lam_l,lam_u,tol,
                               K=100,mode="precision",
                               size=1,info=True)

# CG sampler
# Set random seed
np.random.seed(94)

# Build the matrix A
d = 100
A = np.zeros((d,d))
a = 1.5
eps = 1e-6
s = np.linspace(-3,3,num=d)
for i in range(d):
    for j in range(d):
        if i == j:
            A[i,j] = 2 + eps
        else:
            A[i,j] = 2 * np.exp(-(s[i]-s[j])**2 / (2 * a**2))

invA = np.linalg.inv(A) 
def B(x):
    return A.dot(x)
tol = 1e-2
K = d
size = 5000
init = np.zeros((d,size))
mu = np.zeros((d,size))
theta = np.zeros((d,size))
# Initialization
k = 1
#r_old = np.random.normal(0,1,size=(d,size)) - A(init)
rd = np.random.randint(0, 2, (d, size))
rd[rd == 0] = -1
r_old = rd - B(init)
p_old = r_old
d_old = (p_old * B(p_old)).sum(axis=0)
y = init
omega = init
r_new = np.ones((d,size))
loss_conj = 0
while (col_vector_norms(r_new,2) >= tol).any() and k <= K:
    gam = (r_old * r_old).sum(axis=0) / d_old
    r_new = r_old - gam * B(p_old)
    beta = - (r_new * r_new).sum(axis=0) / (r_old * r_old).sum(axis=0)
    p_new = r_new - beta * p_old
    if (np.abs((p_new * B(p_old)).sum(axis=0)) >= 1e-4).any() and loss_conj == 0:
        print('Loss of conjugacy happened at iteration k = %i.'%k)
        loss_conj = 1
        k_loss = k
    d_new = (p_new * B(p_new)).sum(axis=0)
    z = np.random.normal(0,1,size=size)
    y = y + (z / np.sqrt(d_old)) * B(p_old)
    r_old = r_new
    p_old = p_new
    d_old = d_new
    k = k + 1
theta = mu + y

plt.figure()
plt.imshow(np.cov(theta))
plt.colorbar()

plt.figure()
plt.imshow(A)
plt.colorbar()


## PO sampler
d = 15
mu1 = np.zeros((d,))
mu2 = np.zeros((d,))
K = 15
init = np.zeros((d,))
Lamb1 = np.random.normal(4,1,d)
g =  np.reshape(np.random.normal(4,1,d),(d,1))
M = 1
N = d
Q21 = np.diag(np.random.normal(4,1,d))
b2 = 0

C1 = circulant(g)
Q1 = np.dot(C1.T,np.dot(np.diag(1/Lamb1),C1)) 
Q1 = Q1 + Q21

size = 10000

eta1 = np.reshape(mu1,(d,1)) \
      + np.random.normal(0,1,size=(d,size)) * np.sqrt(Lamb1)[:, np.newaxis]
[eta2,C] = sampler_band(mu2,Q21,b2,mode="precision",size=size)
M = M
N = N
        
def Q2(x):
    CTx = np.zeros((M*N,size))
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
        
theta = CG(Q,eta,x=None,K=None,tol=None,info=False)

S = sampler_PO(mu1,mu2,K,init,size=10000)
theta = S.circu_diag_band(Lamb1,g,M,N,Q21,b2)
Q_hat = EmpiricalCovariance().fit(theta.T).precision_

### MS sampler
import numpy as np
from scipy.linalg import circulant
d = 15
mu1 = np.zeros((d,))
mu2 = np.zeros((d,))
K = 15
init = np.zeros((d,))
Lamb1 = 10 * np.eye(d)
g =  np.reshape(np.random.normal(2,0.1,d),(d,1))
Q21 = np.diag(np.random.normal(0.5,0.1,d))
C1 = circulant(g)
Q1 = np.dot(C1.T,C1) /10
Q = Q1 + Q21 + 100*np.eye(d)
d = len(mu1)
size = 10000
theta = np.zeros((d,size))
        

import mcmc_sampling
S = mcmc_sampling.sampler_MS(mu1,Q,size=10000)
theta = S.exact_MS(method="Jacobi")
theta = S.approx_MS(method="Clone-MCMC",omega=100)

### DA sampler
import mcmc_sampling as mcmc
import numpy as np
from scipy.linalg import circulant
from scipy.linalg import solve_triangular, solve_banded
d = 15
mu1 = np.zeros((d,))
mu2 = np.zeros((d,))
K = 15
init = np.zeros((d,))
Lamb1 = np.random.normal(4,1,d)
g =  np.reshape(np.random.normal(4,1,d),(d,1))
M = 1
N = d
Q21 = np.diag(np.random.normal(4,1,d))
b2 = 0

C1 = circulant(g)
Q = np.dot(C1.T,np.dot(np.diag(1/Lamb1),C1)) 
QQ = Q + Q21

size = 100000

S = mcmc.sampler_DA(mu1,size=size)
theta = S.exact_DA_circu_diag_band(Lamb1,g,M,N,Q21,b2,method="EDA")


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
theta = np.zeros((d,size))
u1 = np.zeros(d)
            
for t in range(size-1):
            
    # Sample the auxiliary variable u2
    mu_u2 = np.fft.ifft(np.fft.fft(u1,axis=0) \
                        * np.reshape(LambG1,(d,)),axis=0).real
    u2 = mu_u2 + np.random.normal(0,1,size=d) * np.sqrt(1/Lamb1)
            
    # Sample the auxiliary variable u1
    mu_u1 = theta[:,t] \
    - omega * np.reshape(Q1(np.reshape(theta[:,t],(d,1))),(d,)) \
    + omega * np.fft.ifft(np.reshape(LambG1.conj(),(d,)) \
                    * np.fft.fft(1/Lamb1 * u2,axis=0),axis=0).real
    u1 = mu_u1 + np.random.normal(0,1,size=d) * np.sqrt(omega)
                
    # Sample the variable of interest theta
    Q_theta = np.eye(d) / omega + Q21
    z = sampler_band(np.zeros(d),Q_theta,b2,mode="precision",
                                        size=1)[0]
    C = sampler_band(np.zeros(d),Q21,b2,mode="precision",
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
               + np.reshape(Q1(np.reshape(mu1,(d,1))),(d,)) \
               + Q2_fun(mu1)
    ab = diagonal_form(Q_theta,lower=b2,upper=b2)
    theta[:,t+1] = solve_banded((b2, b2), ab, mu_theta) \
                               + np.reshape(z,(d,))
                               
# SOR
 
M = np.tril(QQ) \
    + (1-omega)/omega * np.diag(QQ) * np.eye(d)
D = (2-omega)/omega * np.diag(QQ)
def N(x):
    mat = - (np.triu(QQ) - np.diag(QQ) * np.eye(d)) \
    + (1-omega)/omega * np.diag(QQ) * np.eye(d)
    return mat.dot(x)
def NT(x):
    mat = - (np.triu(QQ) - np.diag(QQ) * np.eye(d)) \
    + (1-omega)/omega * np.diag(QQ) * np.eye(d)
    return mat.T.dot(x)
S = np.eye(d) - np.linalg.inv(M.dot(np.diag(1/D)).dot(M.T)).dot(QQ)
lam_l = np.min(np.linalg.eigvals(S)) 
lam_u = np.max(np.linalg.eigvals(S))

theta = np.zeros((d,size))

tau = 2 / (lam_l + lam_u)
beta = tau
alpha = 1
kappa = tau
b = 1
a = (2-tau)/tau + (beta-1) * (1/tau + 1/kappa - 1)
t = 0
# Step 1
z = np.random.normal(0,1,size=d) * np.sqrt(b * D)
Qtheta = N(theta[:,t]) + z
theta_bis = solve_triangular(M,Qtheta,lower=True)
z = np.random.normal(0,1,size=d) * np.sqrt(a * D)
Qtheta = NT(theta_bis) + z
theta[:,t+1] = theta[:,t] \
+ tau * solve_triangular(M.T,Qtheta,lower=False)

# Gibbs sampling
Dinv = 1 / np.diag(QQ)
J = np.eye(d) - QQ * Dinv[:,None]
rho = np.max(np.abs(np.linalg.eigvals(J)))
omega = 2 / (1 + np.sqrt(1 - rho**2))
size = 10000
for t in range(size-2):
    t = t + 1
    beta = 1 / (1/tau - beta * ((lam_u-lam_l)/4)**2)
    alpha = beta/tau
    b = 2*(1-alpha)/alpha * (kappa/tau) + 1
    a = (2-tau)/tau + (beta-1) * (1/tau + 1/kappa - 1)
    z = np.random.normal(0,1,size=d) * np.sqrt(b * D)
    Qtheta = N(theta[:,t]) + z
    theta_bis = solve_triangular(M,Qtheta,lower=True)
    z = np.random.normal(0,1,size=d) * np.sqrt(a * D)
    Qtheta = NT(theta_bis) + z
    theta[:,t+1] = (1 - alpha) * theta[:,t-1] \
    + alpha * theta[:,t]
    + alpha * tau * solve_triangular(M.T,Qtheta,lower=False)
    kappa = alpha * tau + (1 - alpha) * kappa

# Gibbs sampling
Dinv = 1 / np.diag(QQ)
J = np.eye(d) - QQ * Dinv[:,None]
rho = np.max(np.abs(np.linalg.eigvals(J)))
omega = 2 / (1 + np.sqrt(1 - rho**2))
size = 30000
gam = np.sqrt(2 / omega - 1)
theta = np.zeros((d,size))
for t in range(size-1):
    z = np.random.normal(0,1,size=d) * np.sqrt(D)
    Qtheta = N(theta[:,t]) + z
    theta_bis = solve_triangular(M,Qtheta,lower=True)
    z = np.random.normal(0,1,size=d) * np.sqrt(D)
    Qtheta = NT(theta_bis) + z
    theta[:,t+1] = solve_triangular(M.T,Qtheta,lower=False)    

import mcmc_sampling
S = mcmc_sampling.sampler_MS(mu1,QQ,size=30000)
theta = S.exact_MS(method="Gauss-Seidel")
theta = S.approx_MS(method="Clone-MCMC",omega=100)                        
                               