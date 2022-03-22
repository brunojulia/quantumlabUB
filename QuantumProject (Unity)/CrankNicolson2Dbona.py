# Aquest arxiu correspon al codi emprat per resoldre l'equació
# d'Schrödinger 2D dependent del temps

import numpy as np
import math
from time import time




# Aquesta subrutina serveix per resoldre el sistema tridiagonal fent servir
# eliminació de Gauss i substitució inversa.	
def tridiagonal(ds,dc,di,rs):
	# Per fer servir el mètode, hem de considerar que alpha_{N-1}=0 i beta_{N-1}=phi_N
	# Calculem el primer valor fent servir aquestes condicions.
	n = len(rs)
	alpha=np.zeros(n,dtype=complex)
	alpha[0]=(di[0]/dc[0])
	beta=np.zeros(n,dtype=complex)
	beta[0]=rs[0]/dc[0]
	
	for i in range (1,n):
		alpha[i]=(di[i]/(dc[i]-ds[i]*alpha[i-1]))
		beta[i]=(-ds[i]*beta[i-1]+rs[i])/(dc[i]-ds[i]*alpha[i-1])
	#Comencem la sustitució inversa
	vecx=np.zeros(n,dtype=complex)
	vecx[n-1] = beta[n-1]
	
	for j in range (1,n):
		i=(n-1)-j
		vecx[i]=-alpha[i]*vecx[i+1]+beta[i]
		
	return vecx
	

#Aquesta subrutina realitza un pas sencer de Crank-Nicolson ADI des de 
# n fins a n+1.

def PasCrank(psi,dsup,diagox,diagoy,dinf,r,V,dt): 
	# Per passar per tots els punts possibles, primer mantenim la x constant i iterem 
	# sobre totes les y. Repetim el procés per cada x.
	psi_ini=np.copy(psi)
	for i in range(Nx+1):
	# Calculem el vector corresponent als valors de la RHS de la primera equació 
	# del sistema, corresponent als valors per les possibles y d'una mateixa x
	# de n a n+1/2
        	rvec=RHS1(psi[i,:],i,r,V)
	# Resolem el problema tridiagonal amb eliminació de Gauss i substitució inversa.
        	psi[i,:]=tridiagonal(dsup,diagox[i],dinf,rvec)
        	
	# Fem el mateix procés per les x, mantenint y cte i repetint per totes les y
	for j in range(Ny+1):
	# Calculem el vector corresponent als valors de la RHS de la segons equació 
	# del sistema, corresponent als valors per les possibles x d'una mateixa y
	# de n+1/2 a n+1
        	rvec=RHS2(psi[:,j],j,r,V)
	# Resolem el problema tridiagonal amb eliminació de Gauss i substitució inversa.
        	psi_ini[:,j]=tridiagonal(dsup,diagoy[j],dinf,rvec)
        
	return psi_ini


# Aquesta subrutina retorna dos valors diferents segons si es tracta
# de la diagonal, o la diagonal superior o inferior de la matriu tridiagonal.
def Hx(n,r,V):
    H=np.zeros((Nx+1,Nx+1),dtype=complex)
    for i in range(Nx+1):
        for j in range(Nx+1):
            if i==j:
                H[i,j]=1j*((deltat/4.)*V[n,j]+r*2)
            elif abs(i-j)==1:
                H[i,j]=-1j*r
            
                
    return H    

def Hy(n,r,V):
    
    H=np.zeros((Nx+1,Nx+1),dtype=complex)
    for i in range(Nx+1):
        for j in range(Nx+1):
            if i==j:
                H[i,j]=1j*((deltat/4.)*V[i,n]+r*2)
            elif abs(i-j)==1:
                H[i,j]=-1j*r
            
    return H
	
# Aquesta funció dona el valor del vector diagonal de la mariu tridiagonal.

def Adiagx(n,r,V):
    Hamp=Hx(n,r,V)+np.eye(Nx+1)
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if i==j])

def Adiagy(n,r,V):
    Hamp=Hy(n,r,V)+np.eye(Nx+1) 
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if i==j])

# Calcula la diagonal superior i inferior de la matriu tridiagonal.	

def diag_sup_inf(r,V):
    Hamp=Hx(0,r,V)
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if (i-j)==1])



# Aquesta funció serveix per calcular la norma (denistat de probabilitat) de psi	
def norm(psi):
	norma=np.real(psi*np.conj(psi))
	return norma
        
        
 
# Funció d'ona per l'estat fonamental de l'oscil·lador harmònic:
def psi_0_harm(x,y,a,b,m,w,hbar):
    c=((m*w)/(hbar*math.pi))**(1./4.)
    psi0=c*(np.exp(-(m*w*(x**2+y**2))/(2.*hbar))) 
    return psi0
	
# Potencial per a l'oscil·lador harmònic:
def Vh(x,y,xmax,ymax,m,w):
    if x>=xmax or y>=ymax or x<=-xmax or y<=-ymax:
        V=1000000.
    else:
        V=0.5*m*(w**2)*(x**2+y**2)
    return V

# Potencial de partícula en caixa
def Vf(x,y,xmax,ymax,m,w):
    if x>=xmax or y>=ymax or x<=0. or y<=0:
        V=1000000.
    else:
        V=0.
    return V

# Funció d'ona de partícula en caixa
def psill(x,y,a,b,m,w,hbar):
    psil=(math.sqrt(4./(a*b)))*np.sin((np.pi*x)/(a))*np.sin((np.pi*y)/(b))
    return psil

# Mètode dels trapezis per a la resolució d'integrals
def trapezis(dx,fun):
  
    funsum=0.
    for i in range(Nx+1):
        funsum=funsum+(fun[i,0]+fun[i,-1])*((dx**2)/2)
        for j in range(1,Ny):
        	funsum=funsum+fun[i,j]*dx**2
    
    return funsum

# Aplicació del mètode de Crank-Nicolson ADI 2D.
def CrankNicolsonADI_2D (xmin,xmax,ymin,ymax,tmin,tmax,Nx,Ny,Nt,m,V,psi):
	
	# Definició de la malla
    dx=(xmax-xmin)/float(Nx)
    dy=(ymax-ymin)/float(Ny)
    dt=(tmax-tmin)/float(Nt)
	# Creació dels vectors que contenen els punts de la malla
	# on N_k+1 és el nombre de punts en aquella direcció
    x=np.array([xmin+i*dx for i in range(Nx+1)])
    y=np.array([ymin+i*dy for i in range(Ny+1)])
    t=np.array([tmin+i*dt for i in range(Nt+1)])
	# Definim el valor de hbarra en J·s
    hbar=1.
	# Definim el valor de r
    r=(hbar*dt)/(4.*m*((dx)**2))
	# Definim la matriu corresponent al potencial en cada punt
    Vvec=np.array([[Vh(x[i],y[j],xmax,ymax,m,w) for i in range(Nx+1)] for j in range(Ny+1)],dtype=np.float64)
	# Definim el vector corresponent a la diagonal per x i y
    diagx=np.array([Adiagx(i,r,Vvec) for i in range(Nx+1)]) 
    diagy=np.array([Adiagy(i,r,Vvec) for i in range(Ny+1)]) 
    diag_s=np.insert(diag_sup_inf(r,Vvec),0,0) 
    diag_i=np.append(diag_sup_inf(r,Vvec),0)
    
    
    
	# Vector que conté totes les dades (tots els punts a tots els temps):
    psivec=np.zeros((Nx+1,Nx+1,Nt+1),dtype=np.complex128) 
    normes=np.zeros((Nx+1,Nx+1,Nt+1),dtype=np.float64)
	# Definim els vectors per temps inicial
    psivec[:,:,0]=psi
    normes[:,:,0]=norm(psi)
	
	# Hem d'aplicar Crank Nicolson a cada pas de temps.
    for i in range (Nt):
        psi_nou=PasCrank(psivec[:,:,i],diag_s,diagx,diagy,diag_i,r,Vvec,dt)
	#Assignem el valor obtingut al proper temps
        psivec[:,:,i+1]=psi_nou
        normes[:,:,i+1]=norm(psi_nou)
		
    return psivec,normes,t,Vvec

# Definició de la part dreta de les equacions per als dos passos

def RHS1(psi,n,r,V):
    Hamm=(np.eye(Nx+1))-Hx(n,r,V)
    prod=np.dot(Hamm,psi)
   
    
    return prod

def RHS2(psi,n,r,V):
    Hamm=(np.eye(Nx+1))-Hy(n,r,V)
    prod=np.dot(Hamm,psi)    
    
    return prod

	

	
# Proves per graficar. Partícula en caixa.


xmi=0.
xma=1.
ymi=0.
yma=1.
	
m=1.
hbar=1.
tb=0.5
ta=0.
w=2.
deltax=0.03	
deltay=deltax
deltat=0.01
Nx=int((1.)/deltax)
Ny=Nx
Nt=int((tb-ta)/deltat)
psiini=np.array([[psill(xmi+i*deltax,xmi+j*deltay,xma,xma,m,w,hbar) for i in range(Nx+1)]
	for j in range(Nx+1)],dtype=np.complex128)


Pharm=np.array([[Vf(xmi+i*deltax,xmi+j*deltax,xma,xma,m,w) for i in range(Nx+1)]
	for j in range(Nx+1)])

t_ini=time()
psivector,normalitzacio,tvec,Pvec=CrankNicolsonADI_2D(xmi,xma,xmi,xma,ta,tb,Nx,Nx,Nt,m,
	Pharm,psiini)
t_final=time()



np.save('normaharmdx{}dt{}.npy'.format(deltax,deltat),normalitzacio)	
np.save('psiharmdx{}dt{}.npy'.format(deltax,deltat),psivector)
np.save('tvecharmdx{}dt{}.npy'.format(deltax,deltat),tvec)
np.save('Vvecharmdx{}dt{}.npy'.format(deltax,deltat),Pvec)

print(trapezis(deltax,normalitzacio[:,:,0]))
print(trapezis(deltax,normalitzacio[:,:,40]))


# Col·lapse	
	
