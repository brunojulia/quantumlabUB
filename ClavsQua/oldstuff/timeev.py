"""
Jan Albert Iglesias 03/04/2018

This program computes the time evolution for a given wavefunction
in a given time independent potential.
"""

#The problem is solved only in an infinite square box of specified length.

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


#Parameters:
tf = 4.5 #s
dt = 0.05 #s

#Dimensions of the infinite square box where the problem is solved.
a = -15.
b = 15.

#Number of used intervals.
N = 800
deltax = (b-a)/float(N)

#Only the eigen functions from the ground-state to the Nbasis-state will be used as basis vectors. (0<=Nbasis<=N)
Nbasis=150

xarr = np.arange(a,b+deltax*0.1,deltax)

#Initial wave function.
def psi(x):
    f = np.sqrt((1./(np.sqrt(2*np.pi)))*np.exp(-(x)**2/(2.)))
    return f

#Potential.
def gaussian(x):
    f = 4./np.sqrt(2*np.pi*1**2)*np.exp(-(x)**2/(2.*1**2))
    return f

def harmonic(x):
    V = 0.05*x**2
    return V

def pot(x):
    P = gaussian(x) + harmonic(x)
    return P

#It first uses the code from timeind.py to compute the eigenvalues and eigenvectors for the given potential.
def srindwall(a,b,N,pot):
    """
    It solves the time independent one-dimensional Schrödinger equation for a given potential pot(x).
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
    H[0,0] = 1./deltax**2 + 10000000000000 #We manually add large potential walls at the ends of the box. It cannot be extremely large.
    H[0,1] = -1./(2*deltax**2)
    H[N,N] = 1./deltax**2 + 10000000000000
    H[N,N-1] = -1./(2*deltax**2)

    #Then it goes over the other values of the three diagonals.
    for i in range(1, N, 1):
        H[i,i-1] = -1./(2*deltax**2)
        H[i,i+1] = -1./(2*deltax**2)
        H[i,i] = 1./deltax**2 + pot(a+i*deltax)

    #Diagonalization.
    eigvals, eigfuns = np.linalg.eigh(H) #This returns the sorted eigenvalues and eigenvectors for a hermitian matrix. (The eigenvecors could be complex)

    #Normalization. It integrates the squared module of the eigenfunction using the trapezoidal rule and computes the normalization factor as 1/sqrt(that).
    norms = 1./np.sqrt(deltax*(1.-np.absolute(eigfuns[0,:])**2/2.-np.absolute(eigfuns[N,:])**2/2.)) #np.linalg.eigh() returns normalized vectors in the sense that (a0**2+a1**2+...+an**2)=1.

    #The factors are then multiplied to the eigenvectors.
    normeigfuns = np.zeros(shape=(N+1,N+1))
    for i in range(0,N+1,1):
        normeigfuns[:,i] = norms[i]*eigfuns[:,i]

    return eigvals, normeigfuns


evals, efuns = srindwall(a,b,N,pot)

#Now, having calculated the eigen functions; it computes the "projections" of the given initial wavefunction on the eigen function basis.
#To do so, it integrates (using the trapezoidal rule for the points where the eigenfunctions are known) the product
#of the conjugate complex of psi(x) and each eigen function.
#The integral should be done in the whole real space, but all the eigenfunctions are 0 outside the box, so it is only done within the box.

coefs = np.zeros(shape=(Nbasis+1,1),dtype=complex)
coef_x_efuns = np.zeros(shape=(N+1,Nbasis+1),dtype=complex)
evalsbasis = np.zeros(shape=(Nbasis+1,1))

#It builds an array with the values of the given psi(x) in the grid points.
psivec = psi(xarr)

for j in range(0,Nbasis+1,1):
    prod = np.conjugate(psivec)*efuns[:,j]
    coefs[j] = deltax*(np.sum(prod) - prod[0]/2. - prod[Nbasis]/2.) #Trapezoidal rule.
    coef_x_efuns[:,j] = coefs[j]*efuns[:,j] #Coefficient*eigenfun(x)
    evalsbasis[j] = evals[j] #It uses only the eigenvalues up to the Nbasis one.

#Finally, the time evolution is computed.
def psiev(t):
    exp = np.exp(np.complex(0,1)*(-evalsbasis*t)) #It uses t with hbar units.
    psiev = np.sum(coef_x_efuns*np.transpose(exp),axis=1)
    return psiev


"Creating the animation"
fig = plt.figure()
ax = plt.axes(xlim=(-10,10), ylim=(0,1.5))
ln, = ax.plot([],[],'r-',lw=1)
txt = ax.text(-9.5,0.8,"")
txt2 = ax.text(-9.5,1.0,"")

def init():
    ln.set_data([],[])
    ax.plot(xarr, gaussian(xarr), 'g--')
    ax.plot(xarr, harmonic(xarr), 'b--')
    plt.xlabel("x")

    return ln,

def update(t):
    psievolution = psiev(t)
    ln.set_data(xarr, np.abs(psievolution)**2)
    txt.set_text("t = " + str(t).ljust(4)[:4] + " hbar")
    txt2.set_text("norm = " + str(deltax*np.sum(np.absolute(psievolution)**2)))

    return ln,txt,txt2

ani = FuncAnimation(fig, update, frames=np.arange(0,tf+dt,dt),
      init_func=init, blit=True, interval=80)

plt.show()
