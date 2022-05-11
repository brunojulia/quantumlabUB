import numpy as np
import matplotlib.pyplot as plt

# Parámetros
nx=100            # número de valores (dimensión)
nt=5          # número de pasos de tiempo
lmax=40
lmin=-40
lx=lmax-lmin
dx=lx/(nx-1)     # amplada de cada interval és dx
dt=1           # interval de temps

# Constantes
hbarra=1
m=1
r=(hbarra**2)/(2*m*(dx**2))
mu=0
sigma_phi=1
sigma_pozo=2

# Vector posició -> minicodi 1-xx
xx=np.linspace(lmin,lmax,nx)

# Vector phi inicial -> minicodi 2-phi0
def phi0(nx,xx):
	phi0=1j*np.zeros(nx)
	for i in range(0,nx):
		phi0[i]=(np.exp(-((xx[i])**2)/8.0))/((8*np.pi)**(1.0/2.0))
	return phi0


# Matriu H -> minicodi 3-hamiltonia
def hamiltonia(nx,r):
	H=1j*np.zeros((nx,nx))

	for i in range(0,nx):
		H[i,i]=(2*r)           # diagonal central
		if i != (nx-1):
			H[i,i+1]=(-1)*r	   # diagonal superior
			H[i+1,i]=(-1)*r    # diagonal inferior
	return H

# Matriu B -> minicodi 4-matriuB
def matriuB(nx,dt,H,hbarra):
	matriu_B=1j*np.zeros((nx,nx))

	for i in range(0,nx):
		matriu_B[i,i]=1-(1j*dt/(2*hbarra))*H[i,i]           # diagonal central
		if i != (nx-1):
			matriu_B[i,i+1]=1-(1j*dt/(2*hbarra))*H[i,i+1]	# diagonal superior
			matriu_B[i+1,i]=1-(1j*dt/(2*hbarra))*H[i+1,i]	# diagonal inferior
	return matriu_B

# Vectors abc -> minicodi 5-abc
def abc(nx,dt,H,hbarra):
	a=1j*np.zeros(nx)    # Diagonal Central
	b=1j*np.zeros(nx)    # Diagonal Inferior
	c=1j*np.zeros(nx)    # Diagonal Superior

	for i in range(0,nx):
		b[i]=1+(1j*dt/(2*hbarra))*H[i,i]
		if i != 0:
			a[i]=1+(1j*dt/(2*hbarra))*H[i,i-1]
		if i != (nx-1):
			c[i]=1+(1j*dt/(2*hbarra))*H[i,i+1]

	return a,b,c

# funció tridiagonal -> minicodi 6-tridiag
def tridiag(a,b,c,d):

#FUNCIÓ TRIDIAGONAL
#Introduïm els vectors de les 3 diagonal i el vector d

	n=len(a)   #Nombre de files (equacions) 

	#Creamos 2 vectores para los nuevos coeficientes:
	cp=1j*np.zeros(n)
	dp=1j*np.zeros(n)

	#Modificamos los coeficientes de la primera fila
	# i los guardamos en los nuevos vectores
	cp[0] = c[0]/b[0] 
	dp[0] = d[0]/b[0]

	for i in range(1,n):
		denom=(b[i]-a[i]*cp[i-1])
		cp[i]=c[i]/denom
		dp[i]=(d[i]-a[i]*dp[i-1])/denom

	phi=1j*np.zeros(n)
	phi[n-1]=dp[n-1]

	for j in range(1,n):
		i=(n-1)-j
		phi[i]=dp[i]-cp[i]*phi[i+1]

	return phi

# (7)
def modul(nx,f):
	modul=np.zeros(nx)
	for i in range(0,nx):
		modul=(abs(f))
	return modul

# (81)
def simpson(h,n,f):
	suma=0.0

	#afegim els extrems
	suma=f[0]+f[len(f)-1]

	#paritats
	for i in range(1,n-1):
		if (i%2==0):
			suma=suma+2*f[i]
		else:
			suma=suma+4*f[i]

	suma=(h/float(3))*suma

	return suma

# (9)
def pozo(nx,xx):
	pozo=1j*np.zeros(nx)
	for i in range(0,nx):
		pozo[i]= 3*(1.0/(sigma_pozo*np.sqrt(2.0*np.pi)))*(1-(np.exp(-((xx[i]-mu)/(4*sigma_pozo))**2)))
	return pozo



# PARÁMETRES EVOLUCIÓ
H=hamiltonia(nx,r)
matriu_B=matriuB(nx,dt,H,hbarra)
a,b,c=abc(nx,dt,H,hbarra)

# FUNCIÓ PHI0 INICIAL
phi0=phi0(nx,xx)
phi0_abs=modul(nx,phi0)
norma=simpson(dx,nx,phi0_abs)

# POZO
pozo=pozo(nx,xx)
pozo_abs=modul(nx,pozo)
graf0=phi0_abs+pozo_abs


# REPPRESENTACIÓ GRÀFICA
plt.plot(xx,pozo_abs)
plt.plot(xx,graf0, label=0)
print('t=',0,', norma=',norma)

# EVOLUCIÓ PHI
for t in range(0,nt):

	# part dreta de l'equació d=B*phi
	d=np.dot(matriu_B,phi0)

	# Apliquem condicions de contorn
	d[0]=0
	d[nx-2]=0

	# Calculem
	phi=tridiag(a,b,c,d)

	# Calculem el mòdul i la norma
	phi_abs=modul(nx,phi)
	graf=phi_abs+pozo_abs
	area=simpson(dx,nx,phi_abs)
	temps=t+1

	print('t=',temps,', norma=',area)
	plt.plot(xx,graf, label=temps)
	plt.legend()

	phi0=phi

plt.show()

