# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:54:55 2025

@author: cdrg
"""

from numba import jit
import numpy as np

#===============================================
@jit(nopython=True)
def create_U(sigma_pl, kappa, dt):
    N = len(dt)
    U = np.eye(N)
    h = np.ones(N)
    
    coeff = 1 + kappa/2
    for i in range(1, N):
       h[i] = h[i-1] * (i - coeff) / i 
    
    for i in range(1, N):
        for j in range(0, N - i):
            U[j, j + i] = h[i]
       
    scale_factors = sigma_pl * (dt ** (-kappa / 4))
           
    for j in range(N):
        for k in range(N):
            U[j, k] *= scale_factors[j]
    
    return U.T


#===============================================
@jit(nopython=True)
def forward_substitution(L, b):
    N = len(b)
    x = np.zeros(N)
    
    for i in range(N):
        sum_val = 0.0
        for j in range(i):
            sum_val += L[i, j] * x[j]
        x[i] = (b[i] - sum_val) / L[i, i]
        
    return x


@jit(nopython=True)
def log_likelihood(theta, noise, dt):
    # thata[0]: amp, theta[1]: kappa
    N = len(noise)
    U = create_U(theta[0], theta[1], dt)
    r = noise.ravel()
    
    ln_det_C = 0.0
    
    for i in range(0, N):
        ln_det_C += 2 * np.log(U[i, i])
        
    #U_inv_r = np.linalg.solve(U, r)
    U_inv_r = forward_substitution(U, r)
    
    r_T_C_inv_r = U_inv_r.T @ U_inv_r
    n_logL = ln_det_C + r_T_C_inv_r
    
    return n_logL


#===============================================
@jit(nopython=True)
def nelder_mead(func, x, noise, dt, step=0.1, tol=1e-6, max_itr=60, 
                alpha=1., gamma=2., rho=0.5, sigma=0.5):
    n = len(x)
    simplex = np.zeros((n+1, n))
    f = np.zeros(n+1)
    
    #===============================================
    # init simplex
    simplex[0] = x
    f[0] = func(x, noise, dt)
    
    for i in range(1, n+1):
        simplex[i] = x
        simplex[i, i-1] += step
        
        f[i] = func(simplex[i], noise, dt)
    
    #===============================================
    # optimization
    itrs = 0
    while True:
        itrs += 1
        # sort
        sort_ind = f.argsort()
        simplex = simplex[sort_ind]
        f = f[sort_ind]
        
        #===========================
        """--Termination--"""
        # max iteration
        if max_itr and itrs >= max_itr:
            print("itrs: " + str(itrs))
            print("Terminated due to reaching max iterations: max_itr=" + str(max_itr) + " !")
            return simplex[0], f[0]
        
        # tol
        if np.max(np.abs(f - f[0])) < tol:
            print("itrs: " + str(itrs))
            print("Terminated due to tolerance!")
            return simplex[0], f[0]
        
        #===========================
        # centroid of x_1:n
        x_bar = simplex[:-1].sum(axis=0) / n
        
        #===========================
        # reflection
        x_r = x_bar + alpha * (x_bar - simplex[-1])
        f_r = func(x_r, noise, dt)
        
        #=============================================
        '''--main--'''
        # reflection: between best and sub_worst
        if f[0] <= f_r < f[-2]:
            simplex[-1] = x_r
            f[-1] = f_r
            
        # reflection: best
        elif f_r < f[0]:
            x_e = x_bar + gamma * (x_r - x_bar)
            f_e = func(x_e, noise, dt)
            
            if f_e < f_r:
                simplex[-1] = x_e
                f[-1] = f_e
            
            else:
                simplex[-1] = x_r
                f[-1] = f_r
            
        # reflection: worse than sub_worst
        else:
            x_c = x_bar + rho * (simplex[-1] - x_bar)
            f_c = func(x_c, noise, dt)
            
            if f_c < f[-1]:
                simplex[-1] = x_c
                f[-1] = f_c
            
            else:
                # shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f[i] = func(simplex[i], noise, dt)
                    
