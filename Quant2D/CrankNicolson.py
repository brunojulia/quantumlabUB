
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
from matplotlib import animation

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
    A1=np.full(ndim,-r,dtype=complex)
    A3=np.full(ndim,-r,dtype=complex)
    
    A1[ndim-1]=0.
    A3[0]=0.
    
    #B matrix
    B=np.zeros((ndim,ndim),dtype=complex)
    for i in range(0,ndim):
        for j in range(0,ndim):
            if i==j:
                x=i*dx
                y=j*dx
                B[i,j]=1.-2.*r-1j*dt*V(x,y)/(4.*hbar)
            if i==j+1 or i+1==j:
                B[i,j]=r
    
    norma=[]
#    print(Norm(Probability(psi,ndim),ndim,dx), 0)
    for t in range(0,ntime):
        
        psi=CrankStep(dx,dt,ndim,r,V,psi,A1,A3,B)
        
        prob=Probability(psi,ndim)
#        print(Norm(prob,ndim,dx), t+1)
        norma=np.append(norma,Norm(prob,ndim,dx))
    
    return psi,norma

def CrankStep (dx,dt,ndim,r,V,psi,Asup,Ainf,Bmatrix):    
    "1 timestep evolution"
            
    "Evolve for x"
    for j in range(0,ndim):
        
        #Adiag
        Adiag=[]
        for i in range(0,ndim):
            x=i*dx
            y=j*dx
            Adiag=np.append(Adiag,(1.+2.*r+1j*dt*V(x,y)/(4.*hbar)))
        #Psix
        psix=[]
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
        Adiag=[]
        for j in range(0,ndim):
            x=i*dx
            y=j*dx
            Adiag=np.append(Adiag,(1.+2.*r+1j*dt*V(x,y)/(4.*hbar)))
        #Psix
        psiy=[]
        for j in range(0,ndim):
            psiy=np.append(psiy,psi[i,j])
        #Bmatrix*Psi 
        Bproduct=np.dot(Bmatrix,psiy)
        #Tridiagonal
        psiy=tridiag(Ainf,Adiag,Asup,Bproduct)
        
        #Change the old for the new values of psi
        for j in range(0,ndim):
            psi[i,j]=psiy[j]
            
    return psi

#POTENTIAL
def Pot(x,y):
#    k=0.
    #x0=ndim*dx/2.
    return 0.5*k*((x-2.5)**2+(y-2.5)**2)

#INITIAL FUNCTION
        #x->a
        #y->b
def EigenOsci(x,y):
#    k=20.
    m=1.
    
    a=0
    b=0
    if (k==0.):
        return (2./5.)*np.sin(np.pi*(x-2.5)/5)*np.sin(np.pi*(y-2.5)/5)
    else: 
        w=np.sqrt(k/m)
        zetx=np.sqrt(m*w/hbar)*(x-2.5)
        zety=np.sqrt(m*w/hbar)*(y-2.5)
        Hx=ss.eval_hermite(a,zetx)
        Hy=ss.eval_hermite(b,zety)
        c_ab=(2**(a+b)*np.math.factorial(a)*np.math.factorial(b))**(-0.5)
        cte=(m*w/(np.pi*hbar))**0.5
        return c_ab*cte*np.exp(-(zetx**2+zety**2)/2)*Hx*Hy

def Coherent(x,y):
            #x0,y0 descentering 
    k=1.
    m=1.
    w=np.sqrt(k/m)
    
    x0=0.
    y0=0.
    
    xx=np.sqrt(m*w/hbar)*(x-2.5+x0)
    yy=np.sqrt(m*w/hbar)*(y-2.5+y0)
    
    c=(m*w/(np.pi*hbar))**0.5
    e1=1.  #np.exp(-(a**2+b**2)/2)
    e2=1.  #np.exp(np.sqrt(2*m*w/hbar)*(a*xx+b*yy))
    e3=np.exp(-(xx**2+yy**2)/2)
    return c*e1*e2*e3

#INITIAL PSI
def Psizero(F,ndim):
    psi0=np.zeros((ndim,ndim),dtype=complex)
    for i in range (0,ndim):
        for j in range (0,ndim):
            x=dx*i
            y=dx*j
            psi0[i,j]=F(x,y)
    return psi0

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
    



#FIGURE SHOWING POTENTIAL, NORM EVOLUTION AND INTIAL+FINAL PROBABILITY
def fig(Fu0,V,ndim,ntime):
    "Take and show first and last step of time, potential and norm evolution"
        #Potential matrix
    po=Potential_matrix(V,ndim)
        
        #Initial psi
    psi0=Psizero(Fu0,ndim)
    Ppsi0=Probability(psi0,ndim)
    
        #Final psi
    psi,norma=CrankNicolson(dx,dt,ndim,ntime,r,V,psi0)
    Ppsi=Probability(psi,ndim)
    
        #Print norms        
    print('Norm ini t')
    print(Norm(Ppsi0,ndim,dx))
    print('Norm final t')
    print(Norm(Ppsi,ndim,dx))
    
        #Figures
    plt.figure()
            #Potential (colormap)
    plt.subplot(2,2,1)
    plt.title('Potential')
    plt.imshow(po,cmap="viridis")
    plt.colorbar()
            #Initial psi
    plt.subplot(2,2,2)
    plt.title('Initial')
    plt.imshow(Ppsi0,cmap="viridis")
    plt.axis('off')
            #Final psi
    plt.subplot(2,2,4)
    plt.title('Final')
    plt.imshow(Ppsi,cmap="viridis")
    plt.axis('off')
            #Norm evolution
    plt.subplot(2,2,3)
    plt.title('Evol norma')
    plt.plot(norma)
    
    plt.show()
    return


#GIF OF TIME EVOLUTION
def anim(Fu0,V,ndim,ntime):
    "Crank Nicolson and generating an animation"
    
    fig = plt.figure()
    
    #Initial Psi
    psi=Psizero(Fu0,ndim)
    print(Norm(Probability(psi,ndim),ndim,dx))
    
    #A sup and inf diagonals
    A1=np.full(ndim,-r,dtype=complex)
    A3=np.full(ndim,-r,dtype=complex)
    
    A1[ndim-1]=0.
    A3[0]=0.
    
    #B matrix
    B=np.zeros((ndim,ndim),dtype=complex)
    for i in range(0,ndim):
        for j in range(0,ndim):
            if i==j:
                x=i*dx
                y=j*dx
                B[i,j]=1.-2.*r-1j*dt*V(x,y)/(4.*hbar)
            if i==j+1 or i+1==j:
                B[i,j]=r
    #Animation
    ims = []
    for i in range(ntime):
        psi=CrankStep(dx,dt,ndim,r,V,psi,A1,A3,B)
        im = plt.imshow(Probability(psi,ndim), animated=True)
        ims.append([im])
    
    ani=animation.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=500)
    print(Norm(Probability(psi,ndim),ndim,dx))
#    ani.save('TEvol.gif')
    ani.show()
    return
    
#############################################################################

ndim=n=100
#ntime=100

dx=0.05
dt=0.01

m=1.
hbar=1.
r=1j*dt/(4.*m*dx**2)


#f=fig(EigenOsci,Pot,ndim=100,ntime=100)
#an=anim(EigenOsci,Pot,ndim,ntime=100)

#f=fig(Coherent,Pot,ndim=100,ntime=100)
#an=anim(Coherent,Pot,ndim,ntime=200)

#doc=open('Comportament_norma.dat','w')

#for k in range (0,46,15):
    
#    psi0=Psizero(EigenOsci,ndim)
#    norm0=Norm(Probability(psi0,ndim),ndim,dx)
#    print(norm0)
#      
#    psi,norm=CrankNicolson (dx,dt,ndim,151,r,Pot,psi0)
    
#    for t in range(0,101,20):
#        quocient=norm[t] #/norm0
#        doc.write(str(quocient)+'\t')
    
#    doc.write('\n')
#    print(norm[99])
#doc.close()

doc=open('Comportament_norma.dat','r')

titer=np.linspace(0,100,6)

#K=0
line=doc.readline()
line=line.split()
line0=[]
for item in line:
    line0=np.append(line0,float(item))
    
#K=15
line=doc.readline()
line=line.split()
line15=[]
for item in line:
    line15=np.append(line15,float(item))

#K=30
line=doc.readline()
line=line.split()
line30=[]
for item in line:
    line30=np.append(line30,float(item))

#K=45
line=doc.readline()
line=line.split()
line45=[]
for item in line:
    line45=np.append(line45,float(item))

plt.scatter(titer,line0,s=10,c='b', marker="s", label='k=0')
plt.scatter(titer,line15,s=10, c='r', marker="o", label='k=15')
plt.scatter(titer,line30,s=10,c='y', marker="+", label='k=30')
plt.scatter(titer,line45,s=10,c='g', marker="s", label='k=45')

plt.ylabel('Norm value')
plt.xlabel('nº of time iterations')

plt.title('Time evolution of the norm for different k for (0,0) eigenstates of the harmonic oscillator')

axes = plt.gca()
axes.set_ylim([0.9850,1.0005])

plt.legend(loc='lower left')