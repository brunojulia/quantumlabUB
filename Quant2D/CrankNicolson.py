
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

#TRIDIAGONAL ALGORITHM
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

#CRANK-NICOLSON
def CrankNicolson (dx,dt,ndim,ntime,r,V,psi):
    #A sup and inf diagonals
    Asup=np.full(ndim,-r,dtype=complex)
    Ainf=np.full(ndim,-r,dtype=complex)
    
    Asup[ndim-1]=0.
    Ainf[0]=0.
    
    #B matrix
    Bmatrix=np.zeros((ndim,ndim),dtype=complex)
    for i in range(0,ndim):
        for j in range(0,ndim):
            if i==j:
                x=i*dx
                y=j*dx
                Bmatrix[i,j]=1.-2.*r-1j*dt*V(x,y)/(4.*hbar)
            if i==j+1 or i+1==j:
                Bmatrix[i,j]=r
    
    norm=np.array([])
    "Time evolution"
    for t in range(0,ntime+1):
        
        "Evolve for x"
        for j in range(0,ndim):
            
            #Adiag
            Adiag=np.array([])
            for i in range(0,ndim):
                x=i*dx
                y=j*dx
                Adiag=np.append(Adiag,(1.+2.*r+1j*dt*V(x,y)/(4.*hbar)))
            #Psix
            psix=np.array([])
            for i in range(0,ndim):
                psix=np.append(psix,psi[i,j])
            #Bmatrix*Psi0 
            Bproduct=np.dot(Bmatrix,psix)
            
            #Tridiagonal
            psix=tridiag(Ainf,Adiag,Asup,Bproduct)
            
            #Change the old for the new values of psi
            for i in range(0,ndim):
                psi[i,j]=psix[i]
                
        "Evolve for y"
        for i in range(0,ndim):
            
            #Adiag
            Adiag=np.array([])
            for j in range(0,ndim):
                x=i*dx
                y=j*dx
                Adiag=np.append(Adiag,(1.+2.*r+1j*dt*V(x,y)/(4.*hbar)))
            #Psix
            psiy=np.array([])
            for j in range(0,ndim):
                psiy=np.append(psiy,psi[i,j])
            #Bmatrix*Psi 
            Bproduct=np.dot(Bmatrix,psiy)
            #Tridiagonal
            psiy=tridiag(Ainf,Adiag,Asup,Bproduct)
            
            #Change the old for the new values of psi
            for j in range(0,ndim):
                psi[i,j]=psiy[j]
                
        prob=Probability(psi,ndim)
        norm=np.append(norm,Norm(prob,ndim,dx))
    
    return psi,norm

#POTENTIAL
def Pot(x,y):
    k=1.
    return 0.5*k*((x-2.5)**2+(y-2.5)**2)

#INITIAL FUNCTION
        #x->a
        #y->b
def EigenOsci(x,y,a,b):
    k=1.
    m=1.
    w=np.sqrt(k/m)
    zetx=np.sqrt(m*w/hbar)*(x-2.5)
    zety=np.sqrt(m*w/hbar)*(y-2.5)
    Hx=ss.eval_hermite(a,zetx)
    Hy=ss.eval_hermite(b,zety)
    c_ab=(2**(a+b)*np.math.factorial(a)*np.math.factorial(b)*np.pi)**(-0.5)
    return c_ab*np.exp(-0.5*(zetx**2+zety**2))*Hx*Hy

#PROBABILITY
def Probability(f,n):
    p=np.zeros((n,n),dtype=float)
    for i in range (0,n):
        for j in range (0,n):
            p[i,j]=np.real(np.conjugate(f[i,j])*f[i,j])
    return p

#MATRIX OF POTENTIAL
def Potential_matrix(V,dim):
    pot=np.zeros((dim,dim),dtype=float)
    for i in range (0,dim):
        for j in range (0,dim):
            x=dx*i
            y=dx*j
            pot[i,j]=V(x,y)
    return pot

#NORM
def Norm(probab,dim,pas):
    norm=0.
    for i in range (0,dim):
        for j in range (0,dim):
            norm=norm+probab[i,j]
    return norm*pas**2
    

ndim=n=100

ntime=100

dx=0.05
dt=0.01

m=1.
hbar=1.

r=1j*dt/(4.*m*dx**2)

#Potential matrix
po=Potential_matrix(Pot,ndim)


"Estat fonamental: (0,0)"

#Initial psi
psi0=np.zeros((ndim,ndim),dtype=complex)
for i in range (0,ndim):
    for j in range (0,ndim):
        x=dx*i
        y=dx*j
        psi0[i,j]=EigenOsci(x,y,0,0)
Ppsi0=Probability(psi0,ndim)

#Final psi
psi,norma1=CrankNicolson(dx,dt,ndim,ntime,r,Pot,psi0)
Ppsi=Probability(psi,ndim)


#Figures
print(Norm(Ppsi0,ndim,dx))
print(Norm(Ppsi,ndim,dx))

plt.figure()

#plt.subplot(3,3,1)
#plt.title('Potential')
#plt.imshow(po,cmap="plasma")

plt.subplot(2,3,2)
plt.title('Initial')
plt.imshow(Ppsi0,cmap="plasma")
plt.axis('off')

plt.subplot(2,3,3)
plt.title('Final')
plt.imshow(Ppsi,cmap="plasma")
plt.axis('off')

"Estat (0,1)"

#Initial psi
psi0=np.zeros((ndim,ndim),dtype=complex)
for i in range (0,ndim):
    for j in range (0,ndim):
        x=dx*i
        y=dx*j
        psi0[i,j]=EigenOsci(x,y,0,1)
Ppsi0=Probability(psi0,ndim)

#Final psi
psi,norma2=CrankNicolson(dx,dt,ndim,ntime,r,Pot,psi0)
Ppsi=Probability(psi,ndim)

print(Norm(Ppsi0,ndim,dx))
print(Norm(Ppsi,ndim,dx))

#plt.subplot(2,3,4)
#plt.title('Potential')
#plt.imshow(po,cmap="plasma")

plt.subplot(2,3,5)
plt.title('Initial')
plt.imshow(Ppsi0,cmap="plasma")
plt.axis('off')

plt.subplot(2,3,6)
plt.title('Final')
plt.imshow(Ppsi,cmap="plasma")
plt.axis('off')


    #Evol norma

plt.subplot(2,3,1)
plt.title('Evol norma')
plt.plot(norma1)
plt.subplot(2,3,4)
plt.title('Evol norma')
plt.plot(norma2)