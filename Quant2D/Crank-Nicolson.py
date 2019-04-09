
import numpy as np
import matplotlib.pyplot as plt



#Tridiag function
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


def Vosc(x,y):
    r2=(x-0.5)**2+(y-0.5)**2
    return 0.5*r2
   
        
def psi_ini(x, y):
    """
    Wave function at t = 0
    """
    r2=(x-0.5)**2+(y-0.5)**2
    return (np.pi**(-1/4.))**2*np.exp(-r2)


def Crank_step(psi,Bmatrix,V_pot,ndim,dx,dt,hbar,m):
    "One step of Crank-Nicolson resolution. First evolve in x and then in y"
    r=1j*hbar*dt/(4.*m*dx**2)
              
    "Evolve for x"
    for j in range (0,ndim):
        
        #Amatrix
        Adiag=np.array([])
        for i in range(0,ndim):
            x=i*dx
            y=j*dx
            Adiag=np.append(Adiag,(1.-4.*r+1j*dt*V_pot(x,y)/(2.*hbar)))
                
        Asup=np.full(ndim,r,dtype=complex)
        Ainf=np.full(ndim,r,dtype=complex)
        
        Asup[0]=0.
        Ainf[0]=0.
        
        #psix
        psix=np.array([])
        for i in range(0,ndim):
            psix=np.append(psix,psi[i,j])
        
        #Bproduct
            Bproduct=np.array([])
        for i in range(0,ndim):
            prod=0.
            for k in range(0,ndim):
                prod=prod+Bmatrix[i,k]*psix[i]
            Bproduct=np.append(Bproduct,prod)
                
        #Calculate psix for the next stp of time:
        psix=tridiag(Ainf,Adiag,Asup,Bproduct)
        
        #Change the old for the new values of psi
        for i in range(0,ndim):
            psi[i,j]=psix[i]
    
    
    "Evolve for y"
    for i in range (0,ndim):
        
        #Amatrix
        Adiag=np.array([])
        for j in range(0,ndim):
            x=i*dx
            y=j*dx
            Adiag=np.append(Adiag,(1.-4.*r+1j*dt*V_pot(x,y)/(2.*hbar)))
                
        Asup=np.full(ndim,r,dtype=complex)
        Ainf=np.full(ndim,r,dtype=complex)
        
        Asup[0]=0.
        Ainf[0]=0.
        
        #psiy
        psiy=np.array([])
        for j in range(0,ndim):
            psiy=np.append(psiy,psi[i,j])
        
        #Bproduct
            Bproduct=np.array([])
        for j in range(0,ndim):
            prod=0.
            for k in range(0,ndim):
                prod=prod+Bmatrix[j,k]*psiy[j]
            Bproduct=np.append(Bproduct,prod)
                
        #Calculate psiy for the next stp of time:
        psiy=tridiag(Ainf,Adiag,Asup,Bproduct)
        
        #Change the old for the new values of psi
        for j in range(0,ndim):
            psi[i,j]=psiy[j]
        return psi
    

#det=np.linalg.det(psizero)
#norm=np.append(norm,(abs(det))**2)
def Crank_Nicolson(V_pot,psi0,ntime,ndim,dx,dt,hbar,m):
    "The whole Crank Nicolson evolution through all of the steps of time"
    
    r=1j*dt/(2.*m*dx**2)
    
    #Generate the Bmatrix
    Bmatrix=np.zeros((ndim,ndim),dtype=complex)
    for i in range(0,ndim):
        for j in range(0,ndim):
            if i==j:
                x=i*dx
                y=j*dx
                Bmatrix[i,j]=1.+4.*r-1j*dt*V_pot(x,y)/(2.*hbar)
            if i==j+1 or i+1==j:
                Bmatrix[i,j]=-r 
    
    #"Generate the initial psi tensor"
    psizero=np.zeros((ndim,ndim),dtype=complex)
    for i in range (1,ndim-1):
        for j in range (1,ndim-1):
            x=dx*i
            y=dx*j
            psizero[i,j]=psi0(x,y)
    
    #Crank-Nicolson time evolution
    for k in range (0,ntime):
        steppsi=Crank_step(psizero,Bmatrix,V_pot,ndim,dx,dt,hbar,m)
        
        #Change the old data for the new
        for i in range (1,ndim-1):
            for j in range (1,ndim-1):
                psizero[i,j]=steppsi[i,j]

    return steppsi

def Probability(f,n):
    p=np.zeros((n,n),dtype=float)
    for i in range (0,n):
        for j in range (0,n):
            p[i,j]=abs(f[i,j])**2   
    return p

hbar=1.
m=1.

dt=0.1
ndim=50
ntime=100
Lx=1.
dx=Lx/ndim



psizero=np.zeros((ndim,ndim),dtype=complex)
for i in range (0,ndim):
    for j in range (0,ndim):
        x=dx*i
        y=dx*j
        psizero[i,j]=psi_ini(x,y) 


prob0=Probability(psizero,ndim)

psi=Crank_Nicolson(Vosc,psi_ini,ntime,ndim,dx,dt,hbar=1.,m=1.)
prob=Probability(psi,ndim)

   
#plt.title('t=0')

pot=np.zeros((ndim,ndim),dtype=float)
for i in range (0,ndim):
    for j in range (0,ndim):
        x=dx*i
        y=dx*j
        pot[i,j]=Vosc(x,y) 


plt.figure()

plt.subplot(1,3,1)
plt.title('Pot')
plt.imshow(pot,cmap="plasma")

plt.subplot(1,3,2)
plt.title('t=0')
plt.axis('off')
plt.imshow(prob0,cmap="plasma")

plt.subplot(1,3,3)
plt.title('t=10')
plt.axis('off')
plt.imshow(prob,cmap="plasma")

#plt.legend(('EC','EP','EM'),loc='best')
#plt.subplot(2,2,1)
#plt.plot(t,a[:,0])
#plt.xlabel('t')
#plt.ylabel('x')
#plt.subplot(2,2,4)
#plt.xlabel('x')
#plt.ylabel('y')


#plt.pcolor(densoprob)
