#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:41:50 2022

@author: annamoresoserra
"""
from numba.experimental import jitclass
from numba import int32, float64
import numpy as np


spec = [
    ('n', int32),               
    ('r', float64),
    ('V', float64[:,:]), 
    ('w', float64),       
]


@jitclass(spec)
class Diagonals:

    def __init__(self, n, r, V, w): #omega fora (i r?)
        # object attributes
        self.n = n
        self.r = r
        self.V = V
        self.w = w
      
        
      
    def Hx(self, Nx, Ny, deltat):
        H = np.zeros((Nx+1, Ny+1), dtype = np.complex_)
        for i in range(Nx+1):
            for j in range(Nx+1):
                if i == j:
                    H[i,j] = 1j * ((deltat / (4. * self.w)) * self.V[self.n,j] + self.r*2)
                elif abs(i-j) == 1:
                    H[i,j] = -1j * self.r
        return H 
    
    
    
    def Hy(self, Nx, Ny, deltat):
        H = np.zeros((Nx+1, Ny+1), dtype=np.complex_)
        for i in range(Nx+1):
            for j in range(Nx+1):
                if i == j:
                    H[i,j] = 1j * ((deltat / (4. * self.w)) * self.V[i,self.n] + self.r*2)
                elif abs(i-j) == 1:
                    H[i,j] = -1j * self.r        
        return H
    
    
    
    def Adiagx(self, Nx, Ny, H):
        """This function gives us the value of the diagonal vector of the tridiagonal
        matrix."""
        Hamp = H + np.eye(Nx+1)
        return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Ny+1)
                if i==j])
    
    
    
    def Adiagy(self, Nx, Ny, H):
        Hamp = H + np.eye(Nx+1) 
        return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Ny+1)
                if i==j])
    
    
    
    def diag_sup_inf(self, Nx, Ny, Hx):
        """Allows us to calculate the upper and lower diagonal of the tridiagonal matrix"""
        Hamp = Hx
        return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Ny+1)
                if (i-j)==1])
    
    
    
    def dx(self, psi, Nx, H):
        Hamm = (np.eye(Nx+1)) - H
        return np.dot(Hamm, psi)
    
    
    
    def dy(self, psi, Ny, H):
        Hamm = (np.eye(Ny+1)) - H
        return np.dot(Hamm, psi)
    	
