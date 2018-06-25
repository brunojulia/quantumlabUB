"""
This program solves the time independent one-dimensional Schrödinger equation using two different methods.
"""
import matplotlib.pyplot as plt
import numpy as np


#Used potentials.
def harmpot(x):
    V = 0.5*10*x**2
    return V

def zeropot(x):
    v=0.*x
    return v

def wellpot(x):
    if x < 1 and x > -1:
        V = -5 + 0.*x
    else:
        V=5 + 0.*x
    return V


def srindwall(a,b,N,pot):
    """
    It solves the time independent one-dimensional Schrödinger equation for a given potential pot(x).
    (a,b) define the box where the equation is solved.
    It uses N intervals (N>0).
    This method takes the box where the equation is solved as an infinite square well.
    So a huge potential is defined at both ends.
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
    eigvals, eigfuns = np.linalg.eigh(H) #This returns the sorted eigenvalues and eigenvectors for a hermitian matrix.

    #Normalization. It integrates the squared module of the eigenfunction using the trapezoidal rule and computes the normalization factor as 1/sqrt(that).
    norms = 1./np.sqrt(deltax*(1.-np.absolute(eigfuns[0,:])**2/2.-np.absolute(eigfuns[N,:])**2/2.)) #np.linalg.eigh() returns normalized vectors in the sense that (a0**2+a1**2+...+an**2)=1.

    #The factors are then multiplied to the eigenvectors.
    normeigfuns = np.zeros(shape=(N+1,N+1))
    for i in range(0,N+1,1):
        normeigfuns[:,i] = norms[i]*eigfuns[:,i]

    return eigvals, normeigfuns

def srinddx(a,b,N,pot):
    """
    It solves the time independent one-dimensional Schrödinger equation for a given potential pot(x).
    (a,b) define the box where the equation is solved.
    It uses N intervals (N>0).
    This method actually solves the equation from a+dx to b-dx, in order to improve it at the boundaries.
    """
    deltax = (b-a)/float(N)

    #It is going to be solved within a distance deltax from both ends, i.e. [a+deltax,b-deltax]
    N = N-2
    a = a+deltax
    b = b-deltax

    #It creates a (N+1)x(N+1) matrix representing the Hamiltonian.
    #Its indexes range from 0 to N.
    H = np.zeros(shape=(N+1,N+1))

    #It starts with the first and final values (since H[0,-1] and H[N,N+1] do not exist)
    H[0,0] = 1./deltax**2 + pot(a)
    H[0,1] = -1./(2*deltax**2)
    H[N,N] = 1./deltax**2 + pot(b)
    H[N,N-1] = -1./(2*deltax**2)

    #Then it goes over the other values of the three diagonals.
    for i in range(1, N, 1):
        H[i,i-1] = -1./(2*deltax**2)
        H[i,i+1] = -1./(2*deltax**2)
        H[i,i] = 1./deltax**2 + pot(a+i*deltax)

    #Diagonalization.
    eigvals, eigfuns = np.linalg.eigh(H) #This returns the sorted eigenvalues and eigenvectors for a hermitian matrix.

    #Normalization. It integrates the squared module of the eigenfuntion using the trapezoidal rule and computes the normalization factor as 1/sqrt(that).
    norms = 1./np.sqrt(deltax*(1.-np.absolute(eigfuns[0,:])**2/2.-np.absolute(eigfuns[N,:])**2/2.)) #np.linalg.eigh() returns normalized vectors in the sense that (a0**2+a1**2+...+an**2)=1.

    #The factors are then multiplied to the eigenvectors.
    normeigfuns = np.zeros(shape=(N+1,N+1))
    for i in range(0,N+1,1):
        normeigfuns[:,i] = norms[i]*eigfuns[:,i]

    #We manually add the value at the box ends (equal to 0).
    z = np.zeros(N+1)
    finalnormeigfuns = np.append(np.insert(normeigfuns, 0, z, axis=0), [z], axis=0) #It adds a line of zeros both at the top and at the bottom.

    return eigvals, finalnormeigfuns

#The analytical solution to the infinite square well.
def anali(x,a,b,n):
    if x < b and x > a:
        ff=np.sqrt(2./(b-a))*np.sin(np.pi*n*(x-a)/(b-a))
    else:
        ff=0.*x
    return ff**2


#Representing the harmonic potential.
N = 200
a = -3
b = 3
n = 1 #State
deltax = (b-a)/float(N)
ev01,ef01 = srinddx(a,b+deltax,N,harmpot)
ev,ef = srindwall(a,b,N,harmpot)
potvec = []
for i in np.arange(a,b+deltax,deltax):
    potvec.append(harmpot(i))
plt.plot(np.arange(a,b+deltax,deltax),ef01[:,n-1]**2,'b-',label='Displaced')
plt.plot(np.arange(a,b+deltax,deltax),ef[:,n-1]**2,'r-',label='Usual')
plt.plot(np.arange(a,b+deltax,deltax),potvec,'g-',label='Potential')
plt.axis([a,b,0,2])
plt.legend()
plt.title("Harmonic potential")
plt.show()

#Representing the finite square well.
N = 200
a = -3
b = 3
n = 1 #State
deltax = (b-a)/float(N)
ev01,ef01 = srinddx(a,b+deltax,N,wellpot)
ev,ef = srindwall(a,b,N,wellpot)
potvec = []
for i in np.arange(a,b+deltax,deltax):
    potvec.append(wellpot(i))
plt.plot(np.arange(a,b+deltax,deltax),ef01[:,n-1]**2,'b-',label='Displaced')
plt.plot(np.arange(a,b+deltax,deltax),ef[:,n-1]**2,'r-',label='Usual')
plt.plot(np.arange(a,b+deltax,deltax),potvec,'g-',label='Potential')
plt.axis([a,b,0,2])
plt.legend()
plt.title("Square Well potential")
plt.show()


#Representing the infinite square well.
n = 1 #State.
plt.subplot(221)
plt.title("N = 10")
N = 10
a = 0
b = 1.
deltax = (b-a)/float(N)
ev01,ef01 = srinddx(a,b+deltax,N,zeropot)
ev,ef = srindwall(a,b,N,zeropot)
analivec = []
for i in np.arange(a,b+deltax,deltax):
    analivec.append(anali(i,a,b,n))
plt.plot(np.arange(a,b+deltax,deltax),ef01[:,n-1]**2,'b-',label='Displaced')
plt.plot(np.arange(a,b+deltax,deltax),ef[:,n-1]**2,'r-',label='Usual')
plt.plot(np.arange(a,b+deltax,deltax),analivec,'g-',label='Analytical')
plt.axis([a,b,0,2])
plt.gca().axes.get_xaxis().set_visible(False)
plt.legend()

plt.subplot(222)
plt.title("N = 100")
N = 100
a = 0
b = 1.
deltax = (b-a)/float(N)
ev01,ef01 = srinddx(a,b+deltax,N,zeropot)
ev,ef = srindwall(a,b,N,zeropot)
analivec = []
for i in np.arange(a,b+deltax,deltax):
    analivec.append(anali(i,a,b,n))
plt.plot(np.arange(a,b+deltax,deltax),ef01[:,n-1]**2,'b-',label='Displaced')
plt.plot(np.arange(a,b+deltax,deltax),ef[:,n-1]**2,'r-',label='Usual')
plt.plot(np.arange(a,b+deltax,deltax),analivec,'g-',label='Analytical')
plt.axis([a,b,0,2])
plt.gca().axes.get_xaxis().set_visible(False)
plt.legend()

plt.subplot(223)
plt.title("N = 200")
N = 200
a = 0.
b = 1.
deltax = (b-a)/float(N)
ev01,ef01 = srinddx(a,b+deltax,N,zeropot)
ev,ef = srindwall(a,b,N,zeropot)
analivec = []
for i in np.arange(a,b+deltax,deltax):
    analivec.append(anali(i,a,b,n))
plt.plot(np.arange(a,b+deltax,deltax),ef01[:,n-1]**2,'b-',label='Displaced')
plt.plot(np.arange(a,b+deltax,deltax),ef[:,n-1]**2,'r-',label='Usual')
plt.plot(np.arange(a,b+deltax,deltax),analivec,'g-',label='Analytical')
plt.axis([a,b,0,2])
plt.legend()

plt.subplot(224)
plt.title("N = 400")
N = 400
a = 0.
b = 1.
deltax = (b-a)/float(N)
ev01,ef01 = srinddx(a,b+deltax,N,zeropot)
ev,ef = srindwall(a,b,N,zeropot)
analivec = []
for i in np.arange(a,b+deltax,deltax):
    analivec.append(anali(i,a,b,n))
plt.plot(np.arange(a,b+deltax,deltax),ef01[:,n-1]**2,'b-',label='Displaced')
plt.plot(np.arange(a,b+deltax,deltax),ef[:,n-1]**2,'r-',label='Usual')
plt.plot(np.arange(a,b+deltax,deltax),analivec,'g-',label='Analytical')
plt.axis([a,b,0,2])
plt.legend()

plt.show()
