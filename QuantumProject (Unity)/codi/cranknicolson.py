# Aquest arxiu correspon al codi emprat per resoldre l'equació
# d'Schrödinger 2D dependent del temps

import numpy as np
import math
from time import time
from numba import njit



# Aquesta subrutina serveix per resoldre el sistema tridiagonal fent servir
# eliminació de Gauss i sustitució inversa.	
@njit
def tridiagonal(ds,dc,di,rs):
	# Per fer servir el mètode, hem de considerar que alpha_{N-1}=0 i beta_{N-1}=phi_N
	# Calculem el primer valor fent servir aquestes condicions.
	n = len(rs)
	alpha=np.zeros(n,dtype=np.complex_)
	alpha[0]=(di[0]/dc[0])
	beta=np.zeros(n,dtype=np.complex_)
	beta[0]=rs[0]/dc[0]
	
	for i in range (1,n):
		alpha[i]=(di[i]/(dc[i]-ds[i]*alpha[i-1]))
		beta[i]=(-ds[i]*beta[i-1]+rs[i])/(dc[i]-ds[i]*alpha[i-1])
	#Comencem la sustitució inversa
	vecx=np.zeros(n,dtype=np.complex_)
	vecx[n-1] = beta[n-1]
	
	for j in range (1,n):
		i=(n-1)-j
		vecx[i]=-alpha[i]*vecx[i+1]+beta[i]
		
	return vecx
	

#Aquesta subrutina realitza un pas sencer de Crank-Nicolson ADI des de 
# n fins a n+1.
@njit
def PasCrank(psi,dsup,diagox,diagoy,dinf,r,V,dt): 
	# Per passar per tots els punts possibles, primer mantenim la x constant i iterem 
	# sobre totes les y. Repetim el procés per cada x.
	psi_ini=np.copy(psi)
	for i in range(Nx+1):
	# Calculem el vector corresponent als valors de la RHS de la primera equació 
	# del sistema, corresponent als valors per les possibles y d'una mateixa x
	# de n a n+1/2
        	rvec=dx(psi[i,:],i,r,V)
	# Resolem el problema tridiagonal amb eliminació de Gauss i sustitució inversa.
        	psi[i,:]=tridiagonal(dsup,diagox[i],dinf,rvec)
        	
	# Fem el mateix procés per les x, mantenint y cte i repetint per totes les y
	for j in range(Ny+1):
	# Calculem el vector corresponent als valors de la RHS de la segons equació 
	# del sistema, corresponent als valors per les possibles x d'una mateixa y
	# de n+1/2 a n+1
        	rvec=dy(psi[:,j],j,r,V)
	# Resolem el problema tridiagonal amb eliminació de Gauss i sustitució inversa.
        	psi_ini[:,j]=tridiagonal(dsup,diagoy[j],dinf,rvec)
        
	return psi_ini

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
    Vvec=np.array([[V(x[i],y[j],xmax,ymax,m,w) for i in range(Nx+1)] for j in range(Ny+1)],dtype=np.float64)
	# Definim el vector corresponent a la diagonal per x i y
    diagx=np.array([Adiagx(i,r,Vvec) for i in range(Nx+1)]) 
    diagy=np.array([Adiagy(i,r,Vvec) for i in range(Ny+1)]) 
    diag_s=np.insert(diag_sup_inf(r,Vvec),0,0) 
    diag_i=np.append(diag_sup_inf(r,Vvec),0)
    
    
	#Definim la diagonal superior i inferior (diag_s i diag_i respectivament) 
    
    
    
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

@njit
def Hx(n,r,V):
    #Generem matriu Hx
    H=np.zeros((Nx+1,Ny+1),dtype=np.complex_)
    for i in range(Nx+1):
        for j in range(Nx+1):
            if i==j:
                H[i,j]=1j*((deltat/4.)*V[n,j]+r*2)
            elif abs(i-j)==1:
                H[i,j]=-1j*r
            
                
    return H    

@njit
def Hy(n,r,V):
    #Generem matriu Hy
    H=np.zeros((Nx+1,Ny+1),dtype=np.complex_)
    for i in range(Nx+1):
        for j in range(Nx+1):
            if i==j:
                H[i,j]=1j*((deltat/4.)*V[i,n]+r*2)
            elif abs(i-j)==1:
                H[i,j]=-1j*r
            
    return H
	
# Aquesta funció dona el valor del vector diagonal de la mariu tridiagonal.
@njit
def Adiagx(n,r,V):
    Hamp=Hx(n,r,V)+np.eye(Nx+1)
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if i==j])
@njit
def Adiagy(n,r,V):
    Hamp=Hy(n,r,V)+np.eye(Nx+1) 
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if i==j])

# Calcula la diagonal superior i inferior de la matriu tridiagonal.	
@njit
def diag_sup_inf(r,V):
    Hamp=Hx(0,r,V)
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if (i-j)==1])



# Aquesta funció serveix per calcular la norma (denistat de probabilitat) de psi	

@njit
def norm(psi):
	norma=np.real(psi*np.conj(psi))
	return norma
        
        
 

	
# Potencial per a l'oscil·lador harmònic:
@njit
def Vh(x,y,xmax,ymax,m,w):
    if x>=xmax or y>=ymax or x<=-xmax or y<=-ymax:
        V=1000000.
    else:
        V=0.5*m*(w**2)*(x**2+y**2)
    return V

@njit
def Vh_bar(x,y,xmax,ymax,m,w):
    if x>=xmax or y>=ymax or x<=-xmax or y<=-ymax:
        V=1000000.
    if (-1.5<=x<=-0.5) and (-0.5<=y<=0.5):
        V=1000000.
    else:
        V=0.5*m*(w**2)*(x**2+y**2)
    return V

# Potencial de partícula en caixa i barrera

def Vf(x,y,xmax,ymax,m,w):
    if x>=xmax or y>=ymax or x<=-xmax or y<=-ymax:
        V=1000000.
    if (0.<=x<=-1.) and (-3.<=y<=-1. or 1.<=y<=3.):
        V=10000.
    else:
        V=0.
    return V

def Vll(x,y,xmax,ymax,m,w):
    if x>=xmax or y>=ymax or x<=-xmax or y<=-ymax:
        V=10000000000.
    else:
        V=0.
    return V

#def Vbar(x,y,xmax,ymax,m,w):
#    if x>=xmax or y>=ymax or x<=-xmax or y<=-ymax:
#        V=10000000000.
#    if ((-1.4<=x<=-1.) and ((-3.<=y<=-0.5) or (0.5<=y<=3.))):
#        V=1000000.
#    else:
#        V=0.
#    return V

def Vbar(x,y,xmax,ymax,m,w):
    if x>=xmax or y>=ymax or x<=-xmax or y<=-ymax:
        V=10000000000.
    elif ((-1.4<=x<=-1.) and ((-3<=y<=-0.5)or(0.5<=y<=3.))):
        V=10000000000000.
    else:
        V=0.
    return V

#def barrera(x,y,xmax,ymax,m,w):
#    if x>=xmax or y>=ymax or x<=-xmax or y<=-ymax:
#        V=10000000000.
#    elif ((-1<=x<=1) and (-1<=y<=1)):
#        V=1000000000.
#    else:
#        V=0.
#    return V

def barrera(x,y,xmax,ymax,m,w):
    if x>=xmax or y>=ymax or x<=-xmax or y<=-ymax:
        V=10000000000.
    elif ((-1<=x<=1) and (-1<=y<=1)):
        V=7.
    else:
        V=0.
    return V

def escletxa(x,y,xmax,ymax,m,w):
    if x>=xmax or y>=ymax or x<=-xmax or y<=-ymax:
        V=10000000000.
    elif ((-2.<=x<=-1.5) and ((-2.<=y<=-1.5)or(1.5<=y<=2.))):
        V=1000000000.
    elif ((abs(x)>=2) and (abs(y)>=2)):
        V=1000000000.
    else:
        V=0.
    return V

def Vllcreu(x,y,xmax,ymax,m,w):
    if x>=xmax or y>=ymax or x<=-xmax or y<=-ymax:
        V=10000000000.
    elif ((abs(x)>=2) and (abs(y)>=2)):
        V=1000000000.
    else:
        V=0.
    return V

def Vgauss(x,y,xmax,ymax,m,w):
    if x>=xmax or y>=ymax or x<=-xmax or y<=-ymax:
        V=10000000000.
    else:
        V=5*np.exp(-2*(x**2+y**2))
    return V
    
    
    

@njit
def psi_0_harm(x,y,a,b,m,w,hbar):
    c=((m*w)/(hbar*math.pi))**(1./2.)
    psi0=c*(np.exp(-(m*w*((x-1.)**2+(y-0.)**2))/(2.*hbar))) 
    return psi0

#Funció d'ona pels 3 primers estats
@njit
def psi_harm(n,x,y,a,b,m,w,hbar):
    c=(((m*w)/(hbar*math.pi))**(1./2.))/((np.math.factorial(n))*(2.**n))
    if n==1:
        psi0=c*(np.exp(-(m*w*(x**2+y**2))/(2.*hbar)))*4.*x*y*(m*w)/hbar
    if n==2:
        psi0=c*(np.exp(-(m*w*(x**2+y**2))/(2.*hbar)))*(4.*((math.sqrt((m*w)/hbar)*x)**2)-2.)*(4.*((math.sqrt((m*w)/hbar)*y)**2)-2.)
    if n==3:
        psi0=c*(np.exp(-(m*w*(x**2+y**2))/(2.*hbar)))*(8.*((math.sqrt((m*w)/hbar)*x)**3)-12.*x*math.sqrt((m*w)/hbar))*(8.*((math.sqrt((m*w)/hbar)*y)**3)-12.*y*math.sqrt((m*w)/hbar))
    
    return psi0

@njit
def psi_harm1(x,y,a,b,m,w,hbar):
    c=(((m*w)/(hbar*math.pi))**(1./2.))/(2.)
    psi0=c*(np.exp(-(m*w*(x**2+y**2))/(2.*hbar)))*4.*x*y*(m*w)/hbar
    return psi0


@njit
def psi_harm2(x,y,a,b,m,w,hbar):
    c=(((m*w)/(hbar*math.pi))**(1./2.))/(2.*(2.**2))
    psi0=c*(np.exp(-(m*w*((x-0.)**2+(y)**2))/(2.*hbar)))*(4.*((math.sqrt((m*w)/hbar)*(x-0.))**2)-2.)*(4.*((math.sqrt((m*w)/hbar)*(y))**2)-2.)
    return psi0

@njit
def psi_harm3(x,y,a,b,m,w,hbar):
    c=(((m*w)/(hbar*math.pi))**(1./2.))/(6.*(2.**3))
    psi0=c*(np.exp(-(m*w*(x**2+y**2))/(2.*hbar)))*(8.*((math.sqrt((m*w)/hbar)*x)**3)-12.*x*math.sqrt((m*w)/hbar))*(8.*((math.sqrt((m*w)/hbar)*y)**3)-12.*y*math.sqrt((m*w)/hbar))
    return psi0

# Funció d'ona de partícula en caixa
@njit
def psill(x,y,a,b,m,w,hbar):
    psil=(math.sqrt(1./(a*b)))*np.cos((np.pi*(x))/(2*a))*np.cos((np.pi*(y))/(2*b))
    return psil

#sigma=2
@njit
def psigauss(x,y,a,b,m,w,hbar):
    psig=(1./(math.sqrt(2.*math.pi*0.25)))*np.exp((1j/hbar)*(-10.*(x-2.)-0.*(y-0.))-(1./(4.*0.25))*((x-2.)**2+(y-0.)**2))
    return psig




@njit
def psi0f(x,y,a,b,fun,p0x,p0y):
    n=1./((2*np.pi*dispersiox(-a,-b,fun))**(1/2))
    a= n*np.exp(-((x-2)**2+(y-2)**2)/(4.*dispersiox(-a,-b,fun)))*np.exp(1j*p0x*x+1j*p0y*y)
    return a

@njit
def dispersiox(xa,ya,fun):
    #Fun es la norma
    
    #Valor esperat de x:
    fun1=np.array([[fun[i,j]*(xa+dx*j) for j in range(Nx+1)] 
        for i in range(Nx+1)])
    xesp=trapezis(dx,fun1)
    #Valor esperat de x**2:
    fun2=np.array([[fun[i,j]*(xa+dx*j)**2 for j in range(Nx+1)] 
        for i in range(Nx+1)])
    xesp2=trapezis(dx,fun2)
    
    s2=xesp2-xesp**2
    return s2

@njit
def trapezis(dx,fun):
    
  
    funsum=0.
    for i in range(Nx+1):
        funsum=funsum+(fun[i,0]+fun[i,-1])*((dx**2)/2)
        for j in range(1,Ny):
        	funsum=funsum+fun[i,j]*dx**2
    
    return funsum



@njit
def dx(psi,n,r,V):
    Hamm=(np.eye(Nx+1))-Hx(n,r,V)
    prod=np.dot(Hamm,psi)
   
    
    return prod
@njit
def dy(psi,n,r,V):
    Hamm=(np.eye(Nx+1))-Hy(n,r,V)
    prod=np.dot(Hamm,psi)    
    
    return prod

	

	
# Proves per graficar.


L=3.
m=1.
hbar=1.
tb=2.
ta=0.
w=2.
deltax=0.040
deltay=deltax
deltat=0.01
Nx=int((2.*L)/deltax)
Ny=Nx
Nt=int((tb-ta)/deltat)
#psiini=np.array([[psi_0_harm(-L+i*deltax,-L+j*deltay,L,L,m,w,hbar) for i in range(Nx+1)]
#	for j in range(Nx+1)],dtype=np.complex128)


#Pharm=np.array([[Vh(-L+i*deltax,-L+j*deltax,L,L,m,w) for i in range(Nx+1)]
#	for j in range(Nx+1)])

#t_ini=time()
#psivector,normalitzacio,tvec,Pvec=CrankNicolsonADI_2D(-L,L,-L,L,ta,tb,Nx,Nx,Nt,m,
#	Pharm,psiini)
#t_final=time()
#print(t_final)


#np.save('normaharmdx{}dt{}.npy'.format(deltax,deltat),normalitzacio)	
#np.save('psiharmdx{}dt{}.npy'.format(deltax,deltat),psivector)
#np.save('tvecharmdx{}dt{}.npy'.format(deltax,deltat),tvec)
#np.save('Vvecharmdx{}dt{}.npy'.format(deltax,deltat),Pvec)

#print(trapezis(deltax,normalitzacio[:,:,0]))

#Per n=1
psiini=np.array([[psi_harm2(-L+i*deltax,-L+j*deltay,L,L,m,w,hbar) for i in range(Nx+1)]
	for j in range(Nx+1)],dtype=np.complex128)

Pharm=np.array([[Vgauss(-L+i*deltax,-L+j*deltax,L,L,m,w) for i in range(Nx+1)]
	for j in range(Nx+1)])


t_ini=time()
psivector,normalitzacio,tvec,Pvec=CrankNicolsonADI_2D(-L,L,-L,L,ta,tb,Nx,Nx,Nt,m,
	Vgauss,psiini)
t_final=time()


dxdydt=np.array([deltax,deltay,deltat,t_final-t_ini])
print(t_final-t_ini)

np.save('normaoh2vgauss13dx{}dt{}.npy'.format(deltax,deltat),normalitzacio)	
#np.save('psigescletxap0y0p0x12dx{}dt{}.npy'.format(deltax,deltat),psivector)
#np.save('tvecohvh-1dx{}dt{}.npy'.format(deltax,deltat),tvec)
#np.save('Vvecgescletxap0y0p0x12dx{}dt{}.npy'.format(deltax,deltat),Pvec)
#np.save('dadesgvll0.050dx0.01dt.npy',dxdydt)

print(trapezis(deltax,normalitzacio[:,:,21]))





# Col·lapse	
	
