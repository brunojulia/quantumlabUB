"""
Crank-Nicolson method to solve de A*psy(k+1)=B*psy(k)=b problem where A and B are
2D matrix operators and psy is the wave function in two correlative indexs of time
"""

import numpy as np

hbar=1.
m=1.
dx=0.1
dt=0.15
ndim=10

Lx=dx*ndim

q=hbar**2/(2.*m*dx**2)
r=1j*dt/(2.*m*dx**2)

# =============================================================================
    #Potential and initial function
# =============================================================================

def V(x,y):
    return np.exp((x/100)**2+(y/100)**2)

def psi0(x, y):
    """
    Wave function at t = 0
    """
    x0 = 5
    y0 = 0
    s = 1/np.sqrt(2)
    p0x = 100.0/Lx
    p0y = 0.0/Lx
    r2 = (x-x0)**2 + (y-y0)**2
    return np.exp(-1j*(p0x*x + p0y*y))*np.exp(-r2/(4*s**2))/(2*s**2*np.pi)**(.5)


psi=np.zeros((ndim,ndim),dtype=complex)
for i in range (0,ndim):
    for j in range (0,ndim):
        x=dx*i
        y=dx*j
        psi[i,j]=psi0(x,y)

# =============================================================================
def tridiag(a, b, c, d):
    """
    Analogous to the function tridiag.f
    Refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    """
    n = len(a)

    cp = np.zeros(n, dtype = np.complex)
    dp = np.zeros(n, dtype = np.complex)
    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]

    for i in range(1,n):
        m = (b[i]-a[i]*cp[i-1])
        cp[i] = c[i]/m
        dp[i] = (d[i] - a[i]*dp[i-1])/m

    x = np.zeros(n, dtype = np.complex)
    x[n-1] = dp[n-1]

    for j in range(1,n):
        i = (n-1)-j
        x[i] = dp[i]-cp[i]*x[i+1]

    return x
# =============================================================================


# =============================================================================
# Define the two sides of the matrix product
# =============================================================================

#"Define Hamiltonian"
hamil=np.zeros((ndim,ndim))

for i in range (0,ndim):
    for j in range (0,ndim):
        if i==j:
            hamil[i,j]=2.*q+V(i,j)
        if i==j+1:
            hamil[i,j]=-q
        if i==j-1:
            hamil[i,j]=-q

#"Construct A matrix only defining its three diagonals"
Asup=np.zeros(ndim,dtype=complex)
Adiag=np.zeros(ndim,dtype=complex)
Ainf=np.zeros(ndim,dtype=complex)

for i in range (0,ndim):
     for j in range (0,ndim):
        if i==j:
            Adiag[i]=1.+r*hamil[i,j]
        if i==j+1:
            Asup[i]=1.+r*hamil[i,j]
        if i==j-1:
            Asup[i]=1.+r*hamil[i,j]
            
Asup[ndim-1]=0.
Ainf[0]=0.

#Construct vector B*psy=(1-r*hamilt)*psi=b=bproduct for a given x 
i=1

bproduct=np.zeros(ndim,dtype=complex)
for j in range (0,ndim):
    bproduct[j]=psi[i,j]
    for k in range (0,ndim):
        bproduct[j]=bproduct[j]-r*hamil[j,k]*psi[i,j]

#new psi, after one time-step
psinew=tridiag(Asup,Adiag,Ainf,bproduct)