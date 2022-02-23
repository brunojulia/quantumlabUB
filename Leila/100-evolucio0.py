import numpy as np
import matplotlib.pyplot as plt

nx=1000
nt=4
lmax=15
lmin=-15
lx=lmax-lmin
dx=lx/(nx-1)
dt=0.1
hbarra=1
m=1
r=(hbarra**2)/(2*m*(dx**2))

xx=np.linspace(lmin,lmax,nx)

# (2)
def phi0(nx,xx):
	phi0=1j*np.zeros(nx)
	for i in range(0,nx):
		phi0[i]=(np.exp(-((xx[i])**2)/8.0))/((8*np.pi)**(1.0/2.0))
	return phi0


# (3)
def hamiltonia(nx,r):

	H=1j*np.zeros((nx,nx))

	for i in range(0,nx):
		H[i,i]=(2*r)
		if i != (nx-1):
			H[i,i+1]=(-1)*r
			H[i+1,i]=(-1)*r

	return H


# (4)
def matriuB(nx,dt,H,hbarra):

	matriu_B=1j*np.zeros((nx,nx))

	for i in range(0,nx):
		matriu_B[i,i]=1-(1j*dt/(2*hbarra))*H[i,i]
		if i != (nx-1):
			matriu_B[i,i+1]=1-(1j*dt/(2*hbarra))*H[i,i+1]	#Diagonal Superior
			matriu_B[i+1,i]=1-(1j*dt/(2*hbarra))*H[i+1,i]	

	return matriu_B


# La funció matriuA només s'ha creat per veure si la funció abc
# dona els valors correctes.
def matriuA(nx,dt,H,hbarra):

	matriu_A=1j*np.zeros((nx,nx))

	for i in range(0,nx):
		matriu_A[i,i]=1+(1j*dt/(2*hbarra))*H[i,i]
		if i != (nx-1):
			matriu_A[i,i+1]=1+(1j*dt/(2*hbarra))*H[i,i+1]	#Diagonal Superior
			matriu_A[i+1,i]=1+(1j*dt/(2*hbarra))*H[i+1,i]	

	return matriu_A

# (5)
def abc(nx,dt,H,hbarra):
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


# (6)
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

	phi=1j*np.zeros(n)
	phi[n-1]=dp[n-1]

	for j in range(1,n):
		i=(n-1)-j
		phi[i]=dp[i]-cp[i]*phi[i+1]

	return phi

# Esto no lo tengo muy claro
def inici_evolucio(nx,phi0):
	phii = 1j*np.zeros(nx)
	for i in range(1,nx):
		phii[i]=phi0[i]
	return phii


def evolucio(phi0,a,b,c,d):
    phi=tridiag(a,b,c,d)

    return phi

# (7)
def modul(nx,f):
	phi_abs = np.zeros(nx)
	for i in range(0,nx):
		phi_abs[i]=abs(f[i])

	return phi_abs

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

#funció inicial
phi0=phi0(nx,xx)

#
H=hamiltonia(nx,r)
matriu_B=matriuB(nx,dt,H,hbarra)
a,b,c=abc(nx,dt,H,hbarra)

phi_abs=modul(nx,phi0)
area0=simpson(dx,nx,phi_abs)
plt.plot(xx,phi_abs, label=0)
print(0,area0)


for t in range(0,nt):

	# part dreta de l'equació d=B*phi
	d=np.dot(matriu_B,phi0)

	# Apliquem condicions de contorn
	d[0]=d[0]+2.0*r*phi0[0]
	d[nx-2]=d[nx-2]+2.0*r*phi0[nx-2]

#	phii=inici_evolucio(nx,phi0)


	phi=evolucio(phi0,a,b,c,d)
	phi_abs=modul(nx,phi)
	area=simpson(dx,nx,phi_abs)
	temps=t+1

	print(temps,area)
	plt.plot(xx,phi_abs, label=temps)
	plt.legend()

	phi0=phi


plt.show()
