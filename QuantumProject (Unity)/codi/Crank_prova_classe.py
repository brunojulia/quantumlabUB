#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 09:55:34 2022

@author: annamoresoserra
"""
import numpy as np
from diagonals_implementation import Diagonals
from potentials import Potentials
from wave_functions import WaveFunction
from numba import njit


@njit
def norm(psi):
    """This function calculates phi's norm (probability density).

    Args:
        psi (array): wave function

    Returns:
        array: probabilty density
    """
    return np.real(psi * np.conj(psi))


@njit
def tridiagonal(ds, dc, di, rs):
    """This function is used to solve the tridiagonal system using Gauss 
    elimination and inverse sustitution.

    Args:
        ds (array): upper diagonal elements
        dc (array): diagonal elements
        di (array): lower diagonal elements
        rs (array): RHS

    Returns:
        array: _description_
    """

	# Per fer servir el mètode, hem de considerar que alpha_{N-1}=0 i beta_{N-1}=phi_N
	# Calculem el primer valor fent servir aquestes condicions.
    size = len(rs)
    alpha = np.zeros(size, dtype = np.complex_)
    alpha[0] = (di[0] / dc[0])
    beta = np.zeros(size, dtype = np.complex_)
    beta[0] = rs[0] / dc[0]
	
    for i in range(1, size):
        alpha[i] = (di[i] / (dc[i] - ds[i] * alpha[i - 1]))
        beta[i] = (- ds[i] * beta[i-1] + rs[i]) / (dc[i] - ds[i] * alpha[i-1])
	#Comencem la sustitució inversa
    vecx = np.zeros(size, dtype=np.complex_)
    vecx[size-1] = beta[size-1]
	
    for j in range(1, size):
        i = (size - 1) - j
        vecx[i] = - alpha[i] * vecx[i + 1] + beta[i]
		
    return vecx


@njit
def PasCrank(psi, dsup, diagox, diagoy, dinf, r, V, dt, Nx, Ny, w): 
    """This function is used to perform a whole Crank-Nicolson ADI step from n
    to n+1.

    Args:
        psi (array): _description_
        dsup (array): _description_
        diagox (array): _description_
        diagoy (array): _description_
        dinf (array): _description_
        r (float): _description_
        V (array): _description_
        dt (float): _description_

    Returns:
        array: _description_
    """
    psi_ini = np.copy(psi)
	# Per passar per tots els punts possibles, primer mantenim la x constant i iterem 
	# sobre totes les y. Repetim el procés per cada x.
    for i in range(Nx + 1):
	    # Calculem el vector corresponent als valors de la RHS de la primera equació 
	    # del sistema, corresponent als valors per les possibles y d'una mateixa x
	    # de n a n+1/2
        rhs = Diagonals(i, r, V, w)
        Hx = rhs.Hx(Nx,Ny,dt)
        rvec = rhs.dx(psi[i,:], Nx, Hx)

	    # Resolem el problema tridiagonal amb eliminació de Gauss i sustitució inversa.
        psi[i,:] = tridiagonal(dsup, diagox[i], dinf, rvec)  	
	
    # Fem el mateix procés per les x, mantenint y cte i repetint per totes les y
    for j in range(Ny + 1):
	# Calculem el vector corresponent als valors de la RHS de la segons equació 
	# del sistema, corresponent als valors per les possibles x d'una mateixa y
	# de n+1/2 a n+1
        rhs = Diagonals(j, r, V, w)
        Hy = rhs.Hy(Nx,Ny,dt)
        rvec = rhs.dy(psi[:,j], Ny, Hy)
        
	# Resolem el problema tridiagonal amb eliminació de Gauss i sustitució inversa.
        psi_ini[:,j] = tridiagonal(dsup, diagoy[j], dinf, rvec)
    return psi_ini


def CrankNicolsonADI_2D (xmin, xmax, ymin, ymax, tmin, tmax, Nx, Ny, Nt, m, V, w, psi):
    """Crank-Nicolson ADI 2D method.

    Args:
        xmin (float): box lower limit in x-axis
        xmax (float): box upper limit in x-axis
        ymin (float)): box lower limit in y-axis
        ymax (float)): box upper limit in y-axis
        tmin (float)): time minimum
        tmax (float)): time maximum
        Nx (integer): number of points on the x-axis
        Ny (integer): number of points on the y-axis
        Nt (integer): number of points in the time axis
        m (float): mass
        V (function): potential
        psi (array): wave function
    """
	# Defining the mesh
    dx = (xmax-xmin) / float(Nx)
    dy = (ymax-ymin) / float(Ny)
    dt = (tmax-tmin) / float(Nt)
    
    # Creating the arrays that contain the value of the points in the mesh
	# where N_k+1 is the number of points in the k direction.
    x = np.array([xmin + i*dx for i in range(Nx+1)])
    y = np.array([ymin + i*dy for i in range(Ny+1)])
    t = np.array([tmin + i*dt for i in range(Nt+1)])
    

	# Defnining the constant r in order to simplify the code.
    r = dt/(4. * ((dx) ** 2))
    
	# Defining the array which contains the potential at each point.
    
    Vvec=np.zeros((Nx+1,Ny+1), dtype = np.float64)  
    for i in range (Nx+1):
        for j in range (Ny+1):
            P=Potentials(x[i], y[j], xmax, ymax, m, w)
            if V=="Harmonic":
                Vvec[i,j] = P.Vh()
            if V=="Free":
                Vvec[i,j] = P.Vfree()
    
	# Definim el vector corresponent a la diagonal per x i y
    diagx = np.zeros((Nx+1,Ny+1),dtype=np.complex_)
    diagy = np.zeros((Nx+1,Ny+1),dtype=np.complex_)
    for i in range(Nx+1):
        diagonal=Diagonals(i, r, Vvec, w)
        Hx=diagonal.Hx(Nx,Ny,dt)
        diagx[i]=diagonal.Adiagx(Nx,Ny,Hx)
    for j in range(Ny+1):
        diagonal=Diagonals(j, r, Vvec, w)
        Hy=diagonal.Hy(Nx,Ny,dt)
        diagy[j]=diagonal.Adiagy(Nx,Ny,Hy)
    
 
    #Definim la diagonal superior i inferior (diag_s i diag_i respectivament) 
    # Inicialitzem l'objecte diagonal per una n random, ja que no juga cap paper en 
    # en la creació de les diagonals, ho posem pq hem definit així la classe
    # We add a 0 as the first element(since the upper diagonal starts at x+1)
    diagonal=Diagonals(1, r, Vvec, w)
    Hx=diagonal.Hx(Nx, Ny, dt)
    diag_s = np.insert(diagonal.diag_sup_inf(Nx,Ny,Hx), 0, 0) 
    # We add a 0 as the last element(since the lower diagonal starts ends at x-1)
    diag_i = np.append(diagonal.diag_sup_inf(Nx,Ny,Hx), 0)
    
   
	# Vector que conté totes les dades (tots els punts a tots els temps):
    psivec = np.zeros((Nx+1, Nx+1, Nt+1), dtype = np.complex128) 
    normes = np.zeros((Nx+1, Nx+1, Nt+1), dtype = np.float64)
	# Definim els vectors per temps inicial
    psivec[:,:,0] = psi
    normes[:,:,0] = norm(psi)
    

	# Hem d'aplicar Crank Nicolson a cada pas de temps.
    for i in range(Nt):
        psi_nou = PasCrank(psivec[:,:,i], diag_s, diagx, diagy, diag_i, r, Vvec, dt, Nx, Ny, w)
	#Assignem el valor obtingut al proper temps
        psivec[:,:,i+1] = psi_nou
        normes[:,:,i+1] = norm(psi_nou)
        
        
#    np.save('norms_dx{}dt{}V{}.npy'.format(dx, dt, V),normes)	
#    np.save('psi_dx{}dt{}V{}.npy'.format(dx, dt, V),psivec)
#    np.save('t_dx{}dt{}V{}.npy'.format(dx, dt, V),t)
#    np.save('V_dx{}dt{}V{}.npy'.format(dx, dt, V),Vvec)

    return

@njit
def psi_initial(xmin,xmax,ymin,ymax,Nx,Ny,x0,y0,wave_type,sig,px,py):
    # Defining the mesh
    dx = (xmax-xmin) / float(Nx)
    dy = (ymax-ymin) / float(Ny)
        
    # Creating the arrays that contain the value of the points in the mesh
    # where N_k+1 is the number of points in the k direction.
    x = np.array([xmin + i*dx for i in range(Nx+1)])
    y = np.array([ymin + i*dy for i in range(Ny+1)])
    
    # Defining the initial wave function
    psi_ini=np.zeros((Nx+1,Ny+1), dtype=np.complex_)  
    for i in range (Nx+1):
        for j in range (Ny+1):
            wave=WaveFunction(x[i], y[j], x0, y0)
            if wave_type == "0_harmonic":
                psi_ini[i,j]=wave.psi_harm0()
            if wave_type == "1_harmonic":
                psi_ini[i,j]=wave.psi_harm1()
            if wave_type == "2_harmonic":
                psi_ini[i,j]=wave.psi_harm2()
            if wave_type == "gaussian":
                psi_ini[i,j]=wave.psigauss(sig,px,py)
    return psi_ini
    

