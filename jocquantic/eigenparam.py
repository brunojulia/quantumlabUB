#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manu Canals Codina
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#Potential
#For using these in eigemparam they need to be in units eV
def gaussian(mu, sigma, x):
    f = 1./np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2.*sigma**2))
    return f

def harmonic(k, x):
    V = 0.5*k*x**2
    return V

def poten(x):
    pot = (20*gaussian(0, 2, x) + harmonic(0.1, x))  #Potential
    #factor 10 introduced to minimize the influence of the walls, since a 
    #larger potential keeps the main structure of the wave function centered.
    return pot

#Hamiltonian's eigenvalues and vectors.
def eigenparam(a, b, N, m, poten):
    """
    Returns a vector with the eigenvalues(eV) and another with the eigenvectors
    ((Aº·fs)**-1/2) of the quantum hamiltonian with potential [poten(eV)] (each
    columns is an eigenvector with [N] components). 
    It solves the 1D time-independent Schrödinger equation for the given 
    
        H · phi = E · phi  ;  H = -(1/2m)(d**2/dx**2) + poten(x)
                m := mass / hbar**2  [(eV·Aº**2)**-1]
    
    potential (callable poten(x[eV])) inside of a box [a(Aº), b(Aº)], with [N] 
    intervals. 
    """
    deltax = (b-a)/float(N)
    
    #Dividing the ab segment in N intervals leave us with a (N+1)x(N+1) 
    #hamiltonian, where indices 0 and N correspond to the potentials barriers.
    #The hamiltonian operator has 3 non-zero diagonals (the main diagonal, and
    #the ones next to it), with the following elements.
    
    H = np.zeros((N+1,N+1))
    
    H[0,0] = 1./(m*deltax**2) + 1000000000#(eV)
    H[N,N] = 1./(m*deltax**2) + 1000000000
    H[1,0] = -1./(2.*m*deltax**2)
    H[N-1,N] = -1./(2.*m*deltax**2)
    
    for i in range(1, N):
        H[i,i] = 1./(m*deltax**2) + poten(a + deltax*i)
        H[i-1,i] = -1./(2.*m*deltax**2)
        H[i+1,i] = -1./(2.*m*deltax**2)
        
    #Diagonalizing H we'll get the eigenvalues and eigenvectors (evect 
    #in columns).
    evals, evect = np.linalg.eigh(H)
    
    #Normalization. Used trapezoid's method formula, given that sum(evect**2)=1
    factor = np.zeros((N+1)) #Evect will be multiplied by 1/sqrt(factor)
    for col in range(N+1):
        factor[col] = deltax * (1. - evect[0,col]/2. - evect[-1,col]/2.)

    #Normalized vectors
    for col in range(N+1):
        evect[:,col] *= 1/np.sqrt(factor[col])
         
    return evals, evect

def norm(vector, dx):
    """
    Computes the norm of the vector integrating with trapezoids of witdh dx.
    """
    #With dx constant, the following formula can be deduced.
    n = dx * (sum(vector) - vector[0]/2. - vector[-1]/2.)
    return n

#%%
#First run of eigenparam (results used later)
a = -10.  #Aº
b= 10.    #Aº
m = 0.13 #For an electron (eV·Aº**2)**-1/2
N = 100
dx = (b-a)/float(N)
#Mesh for all the plots (includes the boundaries)
mesh = np.linspace(a, b, N+1)

pvals, pvect = eigenparam(a, b, N, m, poten)
#%%

#Projection of the given intial wave function onto the basis {evect}
#Initial wave function

def gpacket(points, sigma0):
    """
    Generates a vector with the values of the initial wave function in the 
    given points(Aº) (psi0 dimension same as points), hera a gaussian packet.
    Wave function introduced here explicitly.
    
    """
    #In this case psi0 is a gaussian packet 
    mu = points[int(len(points)*0.4)] 
    #sigma0 = 0.8
    p0 = 0 #momentum
    
    wavef = np.sqrt((1./(np.sqrt(2*np.pi)*sigma0))*np.exp(-(points-mu)**2/
                     (2.*sigma0**2)))*np.exp(complex(0,-1)*p0*points)
    return wavef

def psi0(points):
    """
    Generates a vector with the values of the initial wave function in the 
    given points(Aº) (psi0 dimension same as points).
    Wave function introduced here explicitly.
    
    """
    #TRIANGLE WF
    waveflist = [p + 5 if p <= 0 else -p + 5 for p in points]
    wavef = np.array(waveflist)

    return wavef/25.

def comp(evect, psi, deltax):
    """
    Given a basis evect and a vector psi, returns a vector with psi's 
    components.
    """
    compo = []
    for ev in np.transpose(evect): #ev: each basis vector
        #for each ev, we integrate ev conjugate * psi = integrand 
        integrand =[np.conjugate(v)*p for v, p in zip(ev, psi)] 
        #integrations by trapezoids
        compo.append(deltax*(sum(integrand)-integrand[0]/2.-integrand[-1]/.2))
        
    return np.array(compo)
#%%
sigma = []
angleE = []
for s in np.arange(0.1, 1.5, 0.2):
    psi = gpacket(mesh,s)
    psicomp = comp(pvect, psi, (b-a)/float(N))
    expect_E = sum([p*np.abs(c)**2 for p,c in zip(pvals, psicomp)])
    sigma.append(s)
    angleE.append(expect_E) 
    
plt.title(r'$\left\langle E \right\rangle \ vs. \sigma $')
plt.yscale('log')
plt.ylabel(r'$\left\langle E \right\rangle \  (eV)$' )
plt.xlabel(r'$\sigma \  (\AA)$')
plt.plot(sigma, angleE) 

  

#print(expect_E)
#plt.plot(mesh, [poten(point) for point in mesh])
#plt.plot(mesh, [expect_E for i in mesh])
#%%
#Animation (using all the components)
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(a, b), ylim=(0, 1))
line, = ax.plot([], [], lw=2, label = '$\Psi(x,t)$')
pot, = ax.plot([], [], lw = 1, label = '$V(x)$')

x = np.linspace(a, b, N+1)
# initialization function (of each frame): plot the background 
def init():
    line.set_data([], []) #reset each frame
    y = poten(x)/10. #Factor so the potential fits the plot
    plt.title('$|\Psi(x,t)|^2$', fontweight= 'bold', fontsize = 20, pad = 10)
    plt.xlabel('$x (\AA)$', fontsize=15)
    plt.ylabel('$|\Psi(x,t)|^2 \ \ (\AA^{-1/2}·fs^{-1/2})$', fontsize=15)
    plt.legend()
    pot.set_data(x, y)
    return line,

# animation function.  This is called sequentially
def fullanimate(t):  
    wf = []
    for i in range(0, N+1): #each component
        acum = 0
        for j in range(0, N+1): #each eigenvector
            acum += psicomp[j]*pvect[i,j]*np.exp(complex(0,-1)*pvals[j]*t/5)
        wf.append((np.abs(acum))**2) #We add the probability
    
    line.set_data(x, wf)
    
    return line,


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, fullanimate, init_func=init, frames=100, 
                                                       interval=100, blit=True)
anim.save('psievo.gif')

plt.show()


#%%

#Truncated alternative
#Lets calculate how far can we cut off comparing tpsi (truncated) with psi0
diff = []
psinorm = norm((np.abs(psi))**2, dx)
for lv in range(0,N+1): #las vector added
    tpsi = []
    for i in range(0, N+1): #each component
        acum = 0
        for j in range(0, lv+1): #each eigenvector up to lv (included)
            acum += psicomp[j] * pvect[i,j]  #t = 0
        tpsi.append(acum)
    #Lets use the normalization condition to compare how much they differ
    diff.append(abs(psinorm - norm((np.abs(tpsi))**2, dx)))
    
plt.title('Diferència(abs) de la norma entre psi truncada i sencera')
plt.yscale('log')
plt.plot(diff)
#Here we see that we hit machine presicion with lv around 40.





#Check pvals (energy values) with its evect component 
#fig, ax1 = plt.subplots()
#
#ax1.plot(pvals, 'b')
#
#ax1.set_yscale('log')
#ax1.set_xlabel('index')
#ax1.set_ylabel('pvals', color='b')
#
#ax2 = ax1.twinx()
#ax2.plot(psicomp, 'r')
#
#ax2.set_ylabel('comp', color = 'r')
#
#fig.tight_layout()
#plt.show()



#%%






#%%
#Animation with psi truncated
top_vect = 7

#Study
studyx = 50
study = []
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(a, b), ylim=(0, 1))
t_line, = ax.plot([], [], lw=2)
f_line, = ax.plot([], [], lw=2)
pot, = ax.plot([], [], lw = 1)

# initialization function (of each frame): plot the background
def init():
    t_line.set_data([], [])
    f_line.set_data([], [])
    x = np.linspace(a, b, N+1)
    y = poten(x)/10.#It's the factor explined in poten, we rescale the 
                    #potential here so it fits in the plot.
    pot.set_data(x, y)
    return t_line, f_line,

# animation function.  This is called sequentially
def trunc_animate(t):  
    x = np.linspace(a, b, N+1)
    
    y = [] #Truncated values
    for i in range(0, N+1): #each component
        acum = complex(0,0)
        for j in range(0, top_vect+1): #each eigenvector up to top_vect (incld)
            acum += psicomp[j]*pvect[i,j]*np.exp(complex(0,-1)*pvals[j]*t/5)
        y.append((np.abs(acum))**2)
        if i == studyx:
            trunc = (np.abs(acum))**2
    t_line.set_data(x, y)
    
    z = [] #Full values
    for i in range(0, N+1): #each component
        acum = complex(0,0)
        for j in range(0, N+1): #each eigenvector
            acum += psicomp[j]*pvect[i,j]*np.exp(complex(0,-1)*pvals[j]*t/5)
        z.append((np.abs(acum))**2)
        if i == studyx:
            study.append([t,np.abs(trunc-(np.abs(acum))**2)])
    
    f_line.set_data(x, z)
    
    
    return t_line, f_line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, trunc_animate, init_func=init, frames=100, 
                                                       interval=100, blit=True)

anim.save('tpsievo.gif')

plt.show()

#%%
