"""
Jan Albert Iglesias 06/05/2018

This program computes the time evolution for a given wavefunction
in a given time independent potential.
"""

import numpy as np

p0 = 0

#Initial wave function.
def psi(pos, x):
    f = np.sqrt((1./(np.sqrt(2*np.pi)*sigma0))*np.exp(-(x-pos)**2/(2.*sigma0**2)))*np.exp(
    complex(0,-1)*p0*x)
    return f

#Potential.
def gaussian(mu, sigma, x):
    f = 1./np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2.*sigma**2))
    return f

def harmonic(k, x):
    V = 0.5*k*x**2
    return V

def pot(mu, sigma, k, x):
    P = 2*factor*(gaussian(mu, sigma, x) + harmonic(k, x))
    return P

def pot1(mu, sigma, k, x):
    P = 2*factor*(np.sqrt(2*np.pi*sigma**2)/10*gaussian(mu, sigma, x) + harmonic(k, x))
    return P


def srindwall(a, b, N, m, pot, mu, sigma, k):
    """
    It solves the time independent one-dimensional Schrödinger equation for a given potential pot(mu, sigma, k, x).
    (a,b) define the box where the equation is solved.
    It uses N intervals (N>0).
    This method takes the box where the equation is solved as an infinite square well.
    So a huge potential is defined at both ends.
    It returns a vector with the normalized eigenvalues and a matrix
    with the eigenvectors, so that eigvec[:,i] is the eigenvector with index i.
    """
    deltax = (b-a)/float(N)

    #It creates a (N+1)x(N+1) matrix representing the Hamiltonian.
    #Its indexes range from 0 to N.
    H = np.zeros(shape=(N+1,N+1))

    #It starts with the first and final values (since H[0,-1] and H[N,N+1] do not exist)
    H[0,0] = 1./(m*deltax**2) + 10000000000000 #We manually add large potential walls at the ends of the box. It cannot be extremely large.
    H[0,1] = -1./(m*2*deltax**2)
    H[N,N] = 1./(m*deltax**2) + 10000000000000
    H[N,N-1] = -1./(m*2*deltax**2)

    #Then it goes over the other values of the three diagonals.
    for i in range(1, N, 1):
        H[i,i-1] = -1./(m*2*deltax**2)
        H[i,i+1] = -1./(m*2*deltax**2)
        H[i,i] = 1./(m*deltax**2) + pot(mu, sigma, k, a+i*deltax)

    #Diagonalization.mu, sigma,
    eigvals, eigfuns = np.linalg.eigh(H) #This returns the sorted eigenvalues and eigenvectors for a hermitian matrix.
    #(The eigenvectors of a hermitian matrix could be complex. However, since H is a symmetric real matrix, a basis of real eigenvectors can always be found.
    #This basis is the one returned by the function. Nevertheless, recall that a global phase does not modify the physics of the problem.)

    #Normalization. It integrates the squared module of the eigenfunction using the trapezoidal rule and computes the normalization factor as 1/sqrt(that).
    norms = 1./np.sqrt(deltax*(1.-np.absolute(eigfuns[0,:])**2/2.-np.absolute(eigfuns[N,:])**2/2.)) #np.linalg.eigh() returns normalized vectors in the sense that (a0**2+a1**2+...+an**2)=1.

    #The factors are then multiplied to the eigenvectors.
    normeigfuns = np.zeros(shape=(N+1,N+1), dtype=complex)
    for i in range(0,N+1,1):
        normeigfuns[:,i] = norms[i]*eigfuns[:,i]

    return eigvals, normeigfuns


def psiev(evalsbasis, coef_x_efuns, t):
    exp = np.exp(np.complex(0,1)*(-evalsbasis*t/hbar))
    psiev = np.sum(coef_x_efuns*np.transpose(exp),axis=1)
    return psiev
