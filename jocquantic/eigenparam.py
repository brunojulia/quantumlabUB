#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manu Canals Codina

Eigenvalues and eigenfunctions of a time-independent potential (Hamiltonian).
"""
#%%
import numpy as np

def poten(x):
    pot = x**2  #Potential 
    return pot

def eigenparam(a, b, N, m, poten):
    """
    Returns the a vector with the eigenvalues and another with the eigenvectors
    (each eigenvector of dimention N). It solves the time-independent 
    Schrödinger equation for a given potentitial [poten] inside of a 
    box [a, b], with [N] intervals. [m] := hbar**2 / m 
    """
    deltax = (b-a)/float(N)
    
    #Dividing the ab segment in N intervals leave us with a (N+1)x(N+1) 
    #hamiltonian, where indices 0 and N correspond to the potentials barriers.
    #The hamiltonian operator has 3 non-zero diagonals (the main diagonal, and
    #the ones next to it), with the following elements.
    
    H = np.zeros((N+1,N+1))
    
    H[0,0] = 1./(m*deltax**2) + 1000000000
    H[N,N] = 1./(m*deltax**2) + 1000000000
    H[1,0] = -1./(2.*m*deltax**2)
    H[N-1,N] = -1./(2.*m*deltax**2)
    
    for i in range(1, N):
        H[i,i] = 1./(m*deltax**2) + poten(a + deltax*i)
        H[i-1,i] = -1./(2.*m*deltax**2)
        H[i+1,i] = -1./(2.*m*deltax**2)
        
    #Diagonalizing H we'll get the eigenvalues and eigenvectors (columns 
    #in evect).
    evals, evect = np.linalg.eigh(H)
    
    #Normalization
    factor = np.zeros((N+1)) #The evect will be multiplied by 1/sqrt(factor)
    #Integrating with trapezoids
    for col in range(N+1):
        for line in range(N):
            #Area of one trapezoid is added every iteration
            factor[col] += ( (np.abs(evect[line,col]))**2 + 
                          (np.abs(evect[line+1,col]))**2 )*deltax / 2.
    
    #Normalized vectors
    for col in range(N+1):
        evect[:,col] *= 1/np.sqrt(factor[col])
        
    
    return evals, evect




#%%
    #Check normalization
pvals, pvect = eigenparam(-5, 5, 50, 1, poten)

N = 50
dx = (10./50.)

norm = np.zeros((N+1))

for col in range(N+1):
    for line in range(N):
        #Area of one trapezoid is added every iteration
        norm[col] += ( (np.abs(pvect[line,col]))**2 + 
                      (np.abs(pvect[line+1,col]))**2 )*dx / 2.

norm




    
    