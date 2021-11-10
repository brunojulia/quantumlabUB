import numpy as np
import matplotlib.pyplot as plt

#Establim els paràmetres del sistema
tmax=10
lx=5      #longitud de x
dx=1       #interval de temps
dt=1       #interval de x
nx=10      #nombre de punts de x
nt=10      #nombre de punts de temps

#Establim els valors de les constants
hbarra=1
m=1
r=(hbarra**2)/(2*m*(dx**2))
p0=200/float(lx)

#Construim un vector amb els valors de x
xx=np.arange(-5.0,5.0,dx)

#Definim la matriu A
matriu_A=1j*np.zeros((nx,nt))

#Contruïm els vectors queque contindrán els valors de les  3 diagonals:
au=1j*np.arange(nx)
a0=1j*np.arange(nx)
ad=1j*np.arange(nx)

#Definim la matriu Hamiltonià H
H=1j*np.zeros((nx,nt))

#Definim el vector potencial
V=1j*np.zeros(nx)
for i in range(0,nx):
	V[i]=42.55*np.exp(- (xx[i]/0.1)**2)/np.sqrt(0.1*np.pi)  #JuliaCabrera
	#introduir del potencial (gaussià)

#Construïm la matriu Hamiltoniana:
for i in range(0,nx):
	if i==0:
		H[i,i]=(2*r+V[i])		#Posició (1,1)
		H[i,i+1]=(-1)*r 		#Posició (1,2)
	elif i==(nx-1):
		H[i,i]=(2*r+V[i])  		#Posició (nx,nx)
		H[i,i-1]=(-1)*r   		#Posició (nx,nx-1)
	else:
		H[i,i]=(2*r+V[i])  		#Diagonal central
		H[i,i+1]=(-1)*r   		#Diagonal superior
		H[i,i-1]=(-1)*r   		#Diagonal inferior

#Omplim ara els vectors de les 3 diagonals i construïm la matriu A
for i in range(0,nx):
	if i==0:
		a0[i]=1+(1j*dt/(2*hbarra))*H[i,i]		#Posició 1 Diag. Central
		au[i]=1+(1j*dt/(2*hbarra))*H[i,i+1]		#Posició 1 Diag. Superior

		matriu_A[i,i]=1+(1j*dt/(2*hbarra))*H[i,i]       #Posició (1,1)
		matriu_A[i,i+1]=1+(1j*dt/(2*hbarra))*H[i,i+1]	#Posició (1,2)

	elif i==(nx-1):
		a0[i]=1+(1j*dt/(2*hbarra))*H[i,i]		#Posició 1 Diag. Central
		ad[i]=1+(1j*dt/(2*hbarra))*H[i,i-1]		#Posició 1 Diag. Inferior

		matriu_A[i,i]=1+(1j*dt/(2*hbarra))*H[i,i]       #Posició (nx,nx)
		matriu_A[i,i-1]=1+(1j*dt/(2*hbarra))*H[i,i-1]	#Posició (nx,nx-1)
	else:
		a0[i]=1+(1j*dt/(2*hbarra))*H[i,i]		#Posició 1 Diag. Central
		au[i]=1+(1j*dt/(2*hbarra))*H[i,i+1]		#Posició 1 Diag. Superior
		ad[i]=1+(1j*dt/(2*hbarra))*H[i,i-1]		#Posició 1 Diag. Inferior

		matriu_A[i,i]=1+(1j*dt/(2*hbarra))*H[i,i]       #Diagonal Central
		matriu_A[i,i+1]=1+(1j*dt/(2*hbarra))*H[i,i+1]	#Diagonal Superior
		matriu_A[i,i-1]=1+(1j*dt/(2*hbarra))*H[i,i-1]   #Diagonal Inferior

#Creem un vector pels valor de la funció d'ona (?¿)
phi_anterior=1j*np.zeros(nx)
phi_posterior=1j*np.zeros(nx)

#Construïm una matriu solucó per guardar els valor de la funció d'ona a cada t.
matriu_sol=1j*np.zeros((nx,nt))

#Imposem el paquet d'ona inicial
for i in range(0,nx):
	phi_anterior=((np.exp(((-1)*(xx[i]-7)**2)/4.0))/((2*np.pi)**(1.0/4.0)))*np.exp(-1j*p0*xx[i])
#JuliaCabrera


#FUNCIÓ TRIDIAGONAL
	#Aquesta funció resol el sistema: matriu_A*x=b
	#Introduïm els vectors de les 3 diagonal i el vector b

def tridiag(a0,au,ad,b):

	n=len(b)   #Nombre de files (equacions) 

	au[0] /= a0[0]  #Posible división por cero
	b[0] /= a0[0]

	for i in range(1,n):
		denom=a0[i]-(ad[i]*au[i-1])
		au[i]/=denom
		if au[i]==0:
			au[i]=0.00000001
			print('error',i)
		b[i]=(b[i]-(ad[i]*b[i-1]))/denom

	#Substitución hacia atrás
	x=[0 for i in range(n)]
	x[-1]=b[-1]

	for i in range(-2,-1-n,-1):
		x[i]=b[i]-au[i]*x[i+1]

	return x

#Definim el vector b
#el vector de b és un vector de totes les posicions per a cada temps
b=1j*np.zeros(nt)

for i in range(0,nt):
	b=np.dot(matriu_A,phi_anterior)
	phi_posterior=tridiag(a0,au,ad,b)

	#guardem phi_posterior a la matriu solució
	matriu_sol[:,i]=phi_posterior*np.conjugate(phi_posterior)
	phi_anterior=phi_posterior













