import numpy as np
import matplotlib.pyplot as plt

nx=4
lmax=7
lmin=-7
lx=lmax-lmin
dx=lx/(nx-1)
dt=1
hbarra=1
m=1
r=(hbarra**2)/(2*m*(dx**2))

xx=np.linspace(lmin,lmax,nx)


def hamiltonia(nx,r):

	H=1j*np.zeros((nx,nx))

	for i in range(0,nx):
		H[i,i]=(2*r)
		if i != (nx-1):
			H[i,i+1]=(-1)*r
			H[i+1,i]=(-1)*r

	return H


def matriuB(nx,dt,r,H,hbarra):

	matriu_B=1j*np.zeros((nx,nx))

	for i in range(0,nx):
		matriu_B[i,i]=1-(1j*dt/(2*hbarra))*H[i,i]
		if i != (nx-1):
			matriu_B[i,i+1]=1-(1j*dt/(2*hbarra))*H[i,i+1]	#Diagonal Superior
			matriu_B[i+1,i]=1-(1j*dt/(2*hbarra))*H[i+1,i]	

	return matriu_B


# La funció matriuA només s'ha creat per veure si la funció abc
# dona els valors correctes.
def matriuA(nx,dt,r,H,hbarra):

	matriu_A=1j*np.zeros((nx,nx))

	for i in range(0,nx):
		matriu_A[i,i]=1+(1j*dt/(2*hbarra))*H[i,i]
		if i != (nx-1):
			matriu_A[i,i+1]=1+(1j*dt/(2*hbarra))*H[i,i+1]	#Diagonal Superior
			matriu_A[i+1,i]=1+(1j*dt/(2*hbarra))*H[i+1,i]	

	return matriu_A

def abc(nx,r,H):
	a=1j*np.zeros(nx)
	b=1j*np.zeros(nx)
	c=1j*np.zeros(nx)

	for i in range(0,nx):
		b[i]=1+(1j*dt/(2*hbarra))*H[i,i]		#Diag. Central
		if i != 0:
			a[i]=1+(1j*dt/(2*hbarra))*H[i,i-1]  #Diag. Inferior
		if i != (nx-1):
			c[i]=1+(1j*dt/(2*hbarra))*H[i,i+1]	#Diag. Superior

	return a,b,c


# vector d inventado para combrobar que la funcion tridiag funciona bien
d=1j*np.ones(nx)

def tridiag(a,b,c,d):

#FUNCIÓ TRIDIAGONAL
#Introduïm els vectors de les 3 diagonal i el vector d

	n=len(a)   #Nombre de files (equacions) 

	#Creamos dos vectores para los nuevos coeficientes:
	cp=1j*np.zeros(n)
	dp=1j*np.zeros(n)

	#Modificamos los coeficientes de la primera fila i los guardamos en los nuevos vectores
	cp[0] = c[0]/b[0] 
	dp[0] = d[0]/b[0]

	for i in range(1,n):
		denom=(b[i]-a[i]*cp[i-1])
		cp[i]=c[i]/denom
		dp[i]=(d[i]-a[i]*dp[i-1])/denom

	x=1j*np.zeros(n)
	x[n-1]=dp[n-1]

	for j in range(1,n):
		i=(n-1)-j
		x[i]=dp[i]-cp[i]*x[i+1]

	return x


H=hamiltonia(nx,r)
matriu_B=matriuB(nx,dt,r,H,hbarra)
matriu_A=matriuA(nx,dt,r,H,hbarra)
a,b,c=abc(nx,r,H)
x=tridiag(a,b,c,d)

print(x)



