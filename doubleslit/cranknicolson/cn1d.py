"""
An implementation of the Crank-Nicolson implicit method for solving
the time dependent Schrödinger's equation in one dimension for an arbitrary V

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

def psi0(x):
    """
    La funció d'ona a t = 0
    """
    x0 = 0
    s = np.sqrt(1/2)
    p0 = 0.0/Lx
    return np.exp(-1j*p0*x)*np.exp(-(x-x0)**2/(4*s**2))/(2*s**2*np.pi)**(.25)

def Vbarrera(x):
    """
    La funció de la barrera de potencial
    """
    return 42.55*np.exp(-x**2/0.1**2)/np.sqrt(.1*np.pi)


def hamiltonian_diagonals(Vi, dx, dt, hbar = 1, m = 1):
    """
    Returns the three diagonals of the Hamiltonian operator in matrix form.
    Parameters:
        Vi : array
            array containing the values of the potential
        dx : scalar
            space interval size
        dt : scalar
            time intervals size
        hbar : scalar, optional
            value for the hbar, defaults to 1
        m : scalar, optional
            value for the mass of the particle, default to 1

    Returns:
        a,b,c : arrays
            the diagonals
    """
    N = len(Vi)

    b = [Vi[i] + hbar**2/(m*dx**2) for i in range(N)]
    c = [-hbar**2/(2.0*m*dx**2) for i in range(N-1)]

    a = np.array([0]+c)
    b = np.array(b)
    c = np.array(c+[0])

    return a, b, c

def A_diagonals(Ha, Hb, Hc, dx, dt, hbar = 1, m = 1):
    """
    A matrix are the coefficients for the tridiagonal problem

    Parameters:
        Ha, Hb, Hc : diagonals of the hamiltonian matrix
        dx : scalar
            space interval size
        dt : scalar
            time intervals size
    Returns:
        a, b, c ; diagonals of the A matrix
    """
    N = len(Hb)

    a = 1.0j*(dt/(2.0*hbar))*Ha
    b = np.ones(N)+1.0j*(dt/(2.0*hbar))*Hb
    c = 1.0j*(dt/(2.0*hbar))*Hc

    return a, b, c

def B_diagonals(Ha, Hb, Hc, dx, dt, hbar = 1, m = 1):
    """
    B is the matrix that multiplies psi in order to obtain the independent
    terms of the tridiagonal problem

    Parameters:
        Ha, Hb, Hc : diagonals of the hamiltonian matrix
    Returns:
        a, b, c ; diagonals of the A matrix
    """
    N = len(Hb)

    a = -1.0j*(dt/(2.0*hbar))*Ha
    b = np.ones(N)-1.0j*(dt/(2.0*hbar))*Hb
    c = -1.0j*(dt/(2.0*hbar))*Hc

    return a, b, c


def dot_tridiagonal_vec(a, b, c, x):
    """
    Dot product of a tridiagonal matrix and a vector
    IMPORTANT: a[0] = 0 and c[N-1] = 0

    Parameters:
        a, b, c: diagonals
        x : vector
    """
    return np.multiply(b,x)+np.multiply(a,np.roll(x,1))+np.multiply(c,np.roll(x,-1))


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

def crank_nicolson1D(x, psi0, V, t0 = 0, tmax = 5, dt = 0.01):
    """
    Runs the Crank-Nicolson method in an infinie well defined by x.

    Parameters:
    x : array
        x-coordinates of the well. x[0] and x[-1] are the last elements where
        psi != 0.
    psi0 : callable
        function that describes the wave function at t0 = 0, Must be of the form
        psi0(x)
    V ; callable
        function that describes the potential. Must be of the form V(x)
    t0 : scalar, optional
        start time of the computation
    tmax : scalar, optional
        end time of the computation
    dt : scalar, optional
        time step size

    Returns:
    psit : 2d array
        2d array such that psit[i] is the wave function at time t0 + i*dt
    times : array
        times[i] = t0 + i*dt
    """

    if not callable(psi0):
        raise RuntimeError("psi0 is not callable.")
    if not callable(V):
        raise RuntimeError("V is not callable.")

    #Number of steps
    iterations = int((tmax-t0)/dt)
    psit = np.zeros([iterations, len(x)], dtype = np.complex128)
    times = []
    dx = x[1]-x[0]

    #1D array containing the wavefunction
    psi = psi0(x)
    #1D array containing the potential
    Vi = V(x)

    #Builds the hamiltonian matrix.
    Ha, Hb, Hc = hamiltonian_diagonals(Vi, dx, dt)
    #Builds the A matrix
    Aa, Ab, Ac = A_diagonals(Ha, Hb, Hc, dx, dt)
    #Builds the B matrix
    Ba, Bb, Bc = B_diagonals(Ha, Hb, Hc, dx, dt)

    for i in range(iterations):
        #Saves the wave function
        psit[i] = psi

        #Solves tridiagonal problem
        b = dot_tridiagonal_vec(Ba, Bb, Bc, psi)
        psi = tridiag(Aa, Ab, Ac, b)

        #Saves time
        times.append(i*dt)

    return psit, np.array(times)

def Vfree(x):
    return 0*x

def Vhooke(x, w = 1, m = 1):
    return 0.5*m*(w*x)**2


if __name__ == '__main__':
    Lx = -10
    N = 300
    dx = 2*Lx/N

    x = np.arange(-Lx, Lx, dx)
    psit, times = crank_nicolson1D(x, psi0, Vhooke, tmax = 2)
    np.savetxt("psit.dat", psit.view(float))
    np.savetxt("times.dat", times)
    np.savetxt("x.dat", x)
    np.savetxt("V.dat", Vfree(x))
