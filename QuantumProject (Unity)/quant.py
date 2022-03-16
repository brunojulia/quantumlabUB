# Aquest arxiu correspon al codi emprat per resoldre l'equació
# d'Schrödinger 2D dependent del temps

import numpy as np
import math
from time import time



# Aquesta subrutina serveix per resoldre el sistema tridiagonal fent servir
# eliminació de Gauss i sustitució inversa.	
def tridiagonal(ds,dc,di,rs):
	# Per fer servir el mètode, hem de considerar que alpha_{N-1}=0 i beta_{N-1}=phi_N
	# Calculem el primer valor fent servir aquestes condicions.
	n = len(rs)
	alpha=np.zeros(n,dtype=complex)
	alpha[0]=-(di[0]/dc[0])
	beta=np.zeros(n,dtype=complex)
	beta[0]=rs[0]/dc[0]
	
	for i in range (1,n):
		alpha[i]=-(di[i]/(dc[i]+ds[i]*alpha[i-1]))
		beta[i]=-(ds[i]*beta[i-1]-rs[i])/(dc[i]+ds[i]*alpha[i-1])
	#Comencem la sustitució inversa
	vecx=np.zeros(n,dtype=complex)
	vecx[n-1] = beta[n-1]
	
	for j in range (1,n):
		i=(n-1)-j
		vecx[i]=alpha[i]*vecx[i+1]+beta[i]
		
	return vecx
	

#Aquesta subrutina realitza un pas sencer de Crank-Nicolson ADI des de 
# n fins a n+1.

def PasCrank(psi,dsup,diagox,diagoy,dinf,r,V,dt):
	# Per passar per tots els punts possibles, primer mantenim la x constant i iterem 
	# sobre totes les y. Repetim el procés per cada x.
	psi_ini=np.copy(psi)
	for k in range(0,Nx+1):
	# Calculem el vector corresponent als valors de la RHS de la primera equació 
	# del sistema, corresponent als valors per les possibles y d'una mateixa x
	# de n a n+1/2
        	rvec=RHS1(psi_ini,k,r,V,dt)
	# Resolem el problema tridiagonal amb eliminació de Gauss i sustitució inversa.
        	psi[k,:]=tridiagonal(dsup,diagox[k],dinf,rvec)
        	
	# Fem el mateix procés per les x, mantenint y cte i repetint per totes les y
	for k in range(0,Ny+1):
	# Calculem el vector corresponent als valors de la RHS de la segons equació 
	# del sistema, corresponent als valors per les possibles x d'una mateixa y
	# de n+1/2 a n+1
        	rvec=RHS2(psi,k,r,V,dt)
	# Resolem el problema tridiagonal amb eliminació de Gauss i sustitució inversa.
        	psi_ini[:,k]=tridiagonal(dsup,diagoy[k],dinf,rvec)
        
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
	r=(1j*hbar*dt)/(4.*m*(dx)**2)
	# Definim la matriu corresponent al potencial en cada punt
	Vvec=np.array([[Vh(x[i],y[j],m,w) for i in range(Nx+1)] for j in range(Ny+1)],dtype=np.float64)
	# Definim el vector corresponent a la diagonal per x i y
	diagx=np.array([Adiag(i,r,Vvec,dt) for i in range(Nx+1)])
	diagy=np.array([Adiag(i,r,Vvec,dt) for i in range(Ny+1)])
	# Definim la diagonal superior i inferior (diag_s i diag_i respectivament)
	diag_s,diag_i=diag_sup_inf(r,Vvec)
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
	

	
# Aquesta funció dona el valor del vector diagonal de la mariu tridiagonal.

def Adiag(n,r,V,det):
	diag=np.zeros((Nx+1),dtype=np.complex128)
	diag=1j*(det/(4.*hbar))*V[n,:]+r*2+1.
	return diag


# Calcula la diagonal superior i inferior de la matriu tridiagonal.	

def diag_sup_inf(r,V):
	sup=np.full(Nx+1,-1j*r,dtype=np.complex128)
	sup[0]=0.
	inf=np.full(Nx+1,-1j*r,dtype=np.complex128)
	inf[Nx]=0.
	return sup,inf






# Aquesta subrutina retirna un vector amb els valors corresponents a 
# la part dreta de la igualtat (2), corresponent al pas n+1/2->n+1

def RHS1(psi,n,r,V,dt):
	rhs=np.zeros((Nx+1),dtype=np.complex128)
	# Considerem els diferents casos possibles
	if n==0:
		rhs=(1.-1j*(dt/(4.*hbar))*V[n,:]-2.*r)*psi[n,:]+r*(psi[1+n,:])
	elif (n>0 and n<(Nx)):
		rhs=(1.-1j*(dt/(4.*hbar))*V[n,:] - 2.*r)*psi[n,:] +r*(psi[-1+n,:]+psi[1+n,:])
	else:
		rhs=(1.-1j*(dt/(4.*hbar))*V[n,:] - 2.*r)*psi[n,:]+r*(psi[-1+n,:])
	return rhs

# Aquesta subrutina retirna un vector amb els valors corresponents a 
# la part dreta de la igualtat (1), corresponent al pas n->n+1/2

def RHS2(psi,n,r,V,dt):
	rhs=np.zeros((Nx+1),dtype=np.complex128)
	# Considerem els diferents casos possibles
	if n==0:
		rhs=(1.-1j*(dt/(4.*hbar))*V[:,n]-2.*r)*psi[:,n]+r*(psi[:,n+1])
	elif (n>0 and n<(Nx)):
		rhs=(1.-1j*(dt/(4.*hbar))*V[:,n] - 2.*r)*psi[:,n] +r*(psi[:,n-1]+psi[:,n+1])
	else:
		rhs=(1.-1j*(dt/(4.*hbar))*V[:,n] - 2.*r)*psi[:,n]+r*(psi[:,n-1])
	return rhs




# Aquesta funció serveix per calcular la norma (denistat de probabilitat) de psi	
def norm(psi):
	norma=np.array([[np.real((psi[i,j])*np.conj(psi[i,j])) for j in range(Nx+1)] 
		for i in range (Nx+1)])
	return norma
        
        
 
# Funció d'ona per l'estat fonamental de l'oscil·lador harmònic:
def psi_0_harm(x,y,m,w,hbar):
	psi0=(np.sqrt((m*w)/(hbar*np.pi)))*(np.exp(-(m*w*(x**2+y**2))/(2.*hbar)))
	return psi0
	
# Potencial per a l'oscil·lador harmònic:
def Vh(x,y,m,w):
	V=0.5*m*(w**2)*(x**2+y**2)
	return V
	
# Proves per graficar.
L=2.
	
m=1.
hbar=1.
tb=0.5
ta=0
w=math.sqrt(3./m)	
deltax=0.03	
deltay=deltax
deltat=0.01
Nx=int((2*L)/deltax)
Ny=Nx
Nt=int((tb-ta)/deltat)
psiini=np.array([[psi_0_harm(-L+i*deltax,-L+j*deltay,m,w,hbar) for i in range(Nx+1)]
	for j in range(Nx+1)],dtype=np.complex128)
Pharm=np.array([[Vh(-L+i*deltax,-L+j*deltax,m,w) for i in range(Nx+1)]
	for j in range(Nx+1)])
t_ini=time()
psivector,normalitzacio,tvec,Pvec=CrankNicolsonADI_2D(-L,L,-L,L,ta,tb,Nx,Nx,Nt,m,
	Pharm,psiini)
t_final=time()


np.save('normaharmdx{}dt{}.npy'.format(deltax,deltat),normalitzacio)	
np.save('psiharmdx{}dt{}.npy'.format(deltax,deltat),psivector)
np.save('tvecharmdx{}dt{}.npy'.format(deltax,deltat),tvec)
np.save('Vvecharmdx{}dt{}.npy'.format(deltax,deltat),Pvec)


# Col·lapse	
	
