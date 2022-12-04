#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 16:18:42 2022

@author: annamoresoserra
"""
import numpy as np
from diagonals_implementation import Diagonals
from potentials import Potentials
from wave_functions import WaveFunction
from Crank_prova_classe import CrankNicolsonADI_2D, psi_initial
from graphs import norm_graph
from tractament_dades import total_norm

# Selecció dels paràmetres
# Compte!! Els potencials estan fets per xmin=-xmax!!!
xmin = -3.
xmax = 3.
x0 = 0.
ymin = -3.
ymax = 3.
y0 = 0.
dx = 0.04
dy = 0.04
dt = 0.01
tmin = 0.
tmax = 2.
Nx = int((xmax-xmin)/dx)
Ny = int((ymax-ymin)/dy)
Nt = int((tmax-tmin)/dt)
m = 1
V = "Harmonic"
w = 2.
wave_type = "gaussian"
sig = 0.2
px = -3.
py = 0.

# Defining the initial wave function
psi = psi_initial(xmin, xmax, ymin, ymax, Nx, Ny, x0, y0, wave_type, sig, px, py)
  
# Solving the 2D Schrödinger equation      
CrankNicolsonADI_2D (xmin, xmax, ymin, ymax, tmin, tmax, Nx, Ny, Nt, m, V, w, psi)

'''
# Norm calculation
total_norm(dx, dt, xmax)
'''
