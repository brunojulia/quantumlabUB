
import numpy as np
#import matplotlib.pyplot as plt


hbar=1.
m=1.
dx=0.1
dt=0.15
ndim=100

Lx=dx*ndim
Ly=dx*ndim

r=1j*dt/(2.*m*dx**2)

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


def V(x,y):
  #  return np.exp(-(x-0.5)**2-(y-0.5)**2)
    if x==Lx/2:
        return 9999999999.
    elif (x or y)==Lx:
        return 9999999999.
    elif (x or y)==0:
        return 9999999999.
    else:
        return 0.
        
def psi0(x, y):
    """
    Wave function at t = 0
    """
    return np.exp(-(x-2*Lx/5)**2/100-(y-Lx/2)**2/100) #/(np.pi*100)**2


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
                
        #Calculate psix for the next stp of time:
        psix=tridiag(Ainf,Adiag,Asup,Bproduct)
        
        #Change the old for the new values of psi
        for j in range(0,ndim):
            psi[i,j]=psiy[j]
        return psi
    


    
#"Generate the initial psi tensor using psi0 function"
psizero=np.zeros((ndim,ndim),dtype=complex)
densoprob=np.zeros((ndim,ndim),dtype=float)

for i in range (0,ndim):
    for j in range (0,ndim):
        x=dx*i
        y=dx*j
        psizero[i,j]=psi0(x,y)
        densoprob[i,j]=abs(psizero[i,j])**2   

#Bmatrix
Bmatrix=np.zeros((ndim,ndim),dtype=complex)
for i in range(0,ndim):
    for j in range(0,ndim):
        if i==j:
            x=i*dx
            y=j*dx
            Bmatrix[i,j]=1.+4.*r-1j*dt*V(x,y)/(2.*hbar)
        if i==j+1 or i+1==j:
            Bmatrix[i,j]=-r    


'Evolve through time and calculate the probability and '
ntime=100

norm=np.array([])
#det=np.linalg.det(psizero)
#norm=np.append(norm,(abs(det))**2)

for k in range (0,ntime):
    steppsi=Crank_step(psizero,Bmatrix,V,ndim,dx=0.1,dt=0.15,hbar=1.,m=1.)
    
    #Change the old for the new data while writing in doc
    for i in range (0,ndim):
        for j in range (0,ndim):
            psizero[i,j]=steppsi[i,j]
            densoprob[i,j]=abs(psizero[i,j])**2
            
#            doc.write(str(densoprob[i,j])+'\t')
 #       doc.write('\n')
    #det=np.linalg.det(steppsi)
   # norm=np.append(norm,(abs(det))**2)
 
 #   doc.write('\n')
 #   doc.write('\n')
    
    
#plt.title('t=0')
#plt.pcolor(densoprob)