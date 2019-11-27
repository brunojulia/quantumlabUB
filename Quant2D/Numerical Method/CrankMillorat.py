import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt

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

def Adiagonals(N,r):
        b = np.full(N, 1 + 2*r)
        a = np.full(N, -r)
        a[N -1] = 0
        c = np.full(N, -r)
        c[0] = 0
        return a, b, c

def Pot(xx,yy):
    k=10
    return 0.5*k*((xx)**2+(yy)**2)

def PsiIni(xx,yy):
    k=10
    m=1.
    hbar=1.
    w=np.sqrt(k/m)
    
    a=0
    b=0
    
    zetx=np.sqrt(m*w/hbar)*xx
    zety=np.sqrt(m*w/hbar)*yy
    Hx=ss.eval_hermite(a,zetx)
    Hy=ss.eval_hermite(b,zety)
    c_ab=(2**(a+b)*np.math.factorial(a)*np.math.factorial(b))**(-0.5)
    cte=(m*w/(np.pi*hbar))**0.5
    
    return c_ab*cte*np.exp(-(zetx**2+zety**2)/2)*Hx*Hy

def Crank():
    N=100
    dx=0.05
    dt=0.01
    hbar=1.
    
    L=dx*N/2.
    r=1j*dt/(4.*dx**2)
    ntime=100
    
    x=np.arange(-L,L,dx)
    meshx , meshy = np.meshgrid(x,x,sparse=True)
        
    potential=Pot(meshx,meshy)
        
 
    psi0=PsiIni(meshx,meshy)
    
    
    #3D array. First index indicates step of time
    psitime = np.zeros([ntime, N, N], dtype = np.complex)
    
    #2D arrays
    psi=psi0[:,:] +0j
    V=potential[:,:]
            
    #A diagonals
    Asup, Adiag, Ainf = Adiagonals(N,r)
    
    #Bmatrix
    B=np.zeros((N,N),dtype=complex)
    for i in range(0,N):
        for j in range(0,N):
            if i==j:
                B[i,j]=1.-2.*r-1j*dt*V[i,j]/(4.*hbar)
            if i==j+1 or i+1==j:
                B[i,j]=r
    
    for t in range(0,ntime):
        
        "Evolve for x"
        for j in range(0,N):
            #Psix
            psix=psi[:,j] +0j
            #Bmatrix*Psi
            Bproduct=np.dot(B,psix)
            #Tridiagonal
            psix=tridiag(Ainf,Adiag+1j*dt*\
                         V[:,j]/(4.*hbar),Asup,Bproduct)
            #Change the old for the new values of psi
            psi[:,j]=psix[:]
                
        "Evolve for y"
        for i in range(0,N):
            #Psiy
            psiy=psi[i,:] +0j
            #Bmatrix*Psi 
            Bproduct=np.dot(B,psiy)
            #Tridiagonal
            psiy=tridiag(Ainf,Adiag+1j*dt*\
                         V[:,j]/(4.*hbar),Asup,Bproduct)
            
            #Change the old for the new values of psi
            psi[i,:]=psiy[:]
        
        psitime[t,:,:]=psi[:,:]
        
    return psitime


psi=Crank()

plt.figure()

def Probability(f,n):
    p=np.zeros((n,n),dtype=float)
    for i in range (0,n):
        for j in range (0,n):
            p[i,j]=np.real(np.conjugate(f[i,j])*f[i,j])
    return p

plt.title('t=100')
plt.imshow(Probability(psi[99,:,:],n=100),cmap="viridis")
plt.colorbar()
