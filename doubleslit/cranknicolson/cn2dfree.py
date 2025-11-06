"""
An implementation of the Crank-Nicolson implicit method for solving
the time dependent Schrödinger's equation in two dimensions for a free particle
(V = 0)

The method uses tridiagonal matrices extensively. They are stored in the
following way:
( b[0] c[0]  0    0    0   ...
( a[1] b[1] c[1]  0    0   ...
(  0   a[2] b[2] c[2]  0   ...
(  0    0   a[3] b[3] c[3] ...

If N is the size of the matrix, we always have:
c[N-1] = 0
a[0] = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from numba import jit

def psi0(x, y):
    """
    Wave function at t = 0
    """
    x0 = 0
    y0 = 0
    s = 0.5
    p0x = 100.0/Lx
    p0y = 100.0/Lx
    r2 = (x-x0)**2 + (y-y0)**2
    return np.exp(-1j*(p0x*x + p0y*y))*np.exp(-r2/(4*s**2))/(2*s**2*np.pi)**(.5)

def Ai_diagonals(N, r):
    """
    N is the size of the matrix (when computing Ax -> Nx and Ay -> Ny)
    """
    b = np.full(N, 1+2*r)
    a = np.full(N, -r)
    a[0] = 0
    c = np.full(N, -r)
    c[-1] = 0

    return a, b, c

@jit
def compute_bx(row, psi, r):
    """
    Row is the row index.
    Returns the vector bx, dimension Nx (independent terms when solving rows)
    """

    bx = np.zeros(Nx, dtype = np.complex128)
    if row != 0:
        bx += r*psi[row-1,:]

    if row != Ny-1:
        bx += r*psi[row+1,:]

    return bx + (1-2*r)*psi[row,:]

@jit
def compute_by(col, psip, r):
    """
    Col is the column index
    Return the vector by, dimension Ny (independent terms when solving columns)
    """

    by = np.zeros(Ny, dtype = np.complex128)
    if col != 0:
        by += r*(psip[:,col-1])
    if col != Nx-1:
        by += r*(psip[:,col+1])

    return by + (1-2*r)*psip[:,col]

@jit
def tridiag(a, b, c, d):
    """
    Analogous to the function tridiag.f
    Refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    """
    n = len(a)

    cp = np.zeros(n, dtype = np.complex128)
    dp = np.zeros(n, dtype = np.complex128)
    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]

    for i in range(1,n):
        m = (b[i]-a[i]*cp[i-1])
        cp[i] = c[i]/m
        dp[i] = (d[i] - a[i]*dp[i-1])/m

    x = np.zeros(n, dtype = np.complex128)
    x[n-1] = dp[n-1]

    for j in range(1,n):
        i = (n-1)-j
        x[i] = dp[i]-cp[i]*x[i+1]

    return x

def crank_nicolson2D(x, y, psi0, t0 = 0, tmax = 5, dt = 0.01, hbar = 1, m = 1):
    """
    Runs the Crank-Nicolson method in an infinie well defined by x.

    Parameters:
    x : array
        x-coordinates of the well. x[0] and x[-1] are the last elements where
        psi != 0.
    psi0 : callable
        function that describes the wave function at t0 = 0, Must be of the form
        psi0(x)
    t0 : scalar, optional
        start time of the computation
    tmax : scalar, optional
        end time of the computation
    dt : scalar, optional
        time step size
    hbar : scalar, optional
        value for the hbar, defaults to 1
    m : scalar, optional
        value for the mass of the particle, default to 1


    Returns:
    psit : 3d array
        3d array such that psit[i] is the wave function at time t0 + i*dt
    times : array
        times[i] = t0 + i*dt
    """

    if not callable(psi0):
        raise RuntimeError("psi0 is not callable.")

    #Number of steps
    iterations = int((tmax-t0)/dt)
    Nx = x.shape[1]
    Ny = x.shape[0]

    psit = np.zeros([iterations, Ny, Nx], dtype = np.complex128)
    times = []
    dx = y[1]-y[0]

    r = 1j*dt*hbar/(4*m*(dx)**2)
    #2D array containing the wavefunction
    psi = psi0(x, y)

    #Builds the A matrices
    Axa, Axb, Axc = Ai_diagonals(Nx, r)
    Aya, Ayb, Ayc = Ai_diagonals(Ny, r)

    for it in range(iterations):
        #Saves the wave function
        psit[it] = psi

        psip = np.zeros(psi.shape, dtype = np.complex128)
        #rows
        for j in range(Ny):
            bx = compute_bx(j, psi, r)
            psip[j,:] = tridiag(Axa, Axb, Axc, bx)

        #columns
        for i in range(Nx):
            by = compute_by(i, psip, r)
            psi[:,i] = tridiag(Aya, Ayb, Ayc, by)

        #Saves time
        times.append(it*dt)

    return psit, np.array(times)

if __name__ == '__main__':
    Lx = 5.0
    Ny = 300
    Nx = 300
    dx = 2*Lx/Nx
    Ly = Ny*dx/2

    x, y = np.meshgrid(np.arange(-Lx, Lx, dx), np.arange(-Ly, Ly, dx))

    starttime = time()
    psit, times = crank_nicolson2D(x, y, psi0, tmax = 3)
    print(time()-starttime)

    print("Saving to file")
    np.save("psit2d.npy", psit)
    np.savetxt("times2d.dat", times)
    np.savetxt("x2d.dat", x)
    np.savetxt("y2d.dat", y)
