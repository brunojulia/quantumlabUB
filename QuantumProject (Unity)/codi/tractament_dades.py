#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: annamoresoserra
"""
import numpy as np

# Norm at a given time
def trapezis(xa, xb, ya, yb, dx, fun):
    Nx = int((xb-xa)/dx)
    Ny = int((yb-ya)/dx)
    funsum = 0.
    for i in range(Nx+1):
        funsum = funsum + (fun[i,0] + fun[i,-1])*(dx**2)/2
        for j in range(1,Ny):
        	funsum = funsum + fun[i,j]*dx**2
    
    return funsum


# Compte!! Programat per xa=-xa
# Norm at all times
def total_norm(dx, dt, L):
    norma = np.load('norms_dx{}dt{}.npy'.format(dx,dt))
    normalitza = np.zeros((len(norma[1,1,:])))
    for j in range(len(norma[1,1,:])):
        normalitza[j] = trapezis(-L, L, -L, L, dx, norma[:,:,j])
    np.save('normatot_dx{}dt{}.npy'.format(dx,dt),normalitza)
    return


# Energy at a given time
def Energy(psi, dx, dy, V):
    # Defining the kinetic energy
    Ec = np.zeros((np.shape(psi)), dtype=complex)
    # Defining the potential energy
    Ep = np.zeros((np.shape(psi))) 
    Nx = np.int(len(psi[0,:])) - 1
    Ny = Nx
    for i in range (1,Nx):
        for j in range (1,Ny):
            Ec[i,j] = -(1./2.)*(((psi[i+1,j] - 2.*psi[i,j] + psi[i-1,j])/(dx**2))+
				((psi[i,j+1]-2.*psi[i,j]+psi[i,j-1])/(dy**2)))
            Ec[i,j] = np.real(Ec[i,j] * np.conj(psi[i,j]))
            Ep[i,j] = psi[i,j] * np.conj(psi[i,j]) * V[i,j]
    return Ec, Ep


# Energy at all times
def total_energy(psi, dx, dy, dt, V):
    Nt = len(psi[1,1,:])
    # Defining the total kinetic energy
    Htc = np.zeros(Nt)
    # Defining the total potential energy
    Htp = np.zeros(Nt)
    Nx = len(psi[0,:,1]) - 1
    Ny = Nx
    
    # Calculating the values of the energy for each time
    for j in range(Nt):
        Ec,Ep = Energy(psi[:,:,j], dx, dy, V)
        sumac=0.
        sumap=0.
        for n in range(Nx+1):
            for k in range(Nx+1):
                        sumac = sumac + Ec[n,k]
                        sumap = sumap + Ep[n,k]
        Htc[j] = sumac * dx**2
        Htp[j] = sumap * dx**2
        
    np.save('K_total_dx{}dt{}.npy'.format(dx,dt), Htc)
    np.save('P_total_dx{}dt{}.npy'.format(dx,dt), Htp)
    
    return


# w marxar amb les unitats????
# Theoretical energy for a 2D harmonic oscilator in the fundamental state
def Energy_harm_0(w):
    Et = w/2.
    return Et 


# Theoretical energy for a 2D harmonic oscilator in the first excited state
def Energy_harm_1(w):
    Et = (3./2.) * w
    return Et


# Theoretical energy for a 2D harmonic oscilator in the second excited state
def Energy_harm_2(w):
    Et = (5./2.) * w
    return Et


# Theoretical energy of a gaussian wave packet
def Energy_gaussian(pox, poy, sig):
    Eg = (1/(2.) * (pox**2 + poy**2 + (1/(2.*(sig))))
    return Eg

	