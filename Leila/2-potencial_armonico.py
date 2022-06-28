import numpy as np
import matplotlib.pyplot as plt

# Parámetros
nx=100            # número de valores (dimensión)
nt=20             # número de pasos de tiempo
lmax=20
lmin=-20
lx=lmax-lmin
dx=lx/(nx-1)     # amplada de cada interval és dx
dt=0.1           # interval de temps
p0=100/float(lx)
omega=0.1
y1=0.5
a=2
b=3

# Constantes
hbarra=1
m=1
r=(hbarra**2)/(2*m*(dx**2))

xe=np.abs(np.sqrt(2*y1/(b*m*omega**2)))

# Vector posició -> minicodi 1-xx
xx=np.linspace(lmin,lmax,nx)

# Funció que ens retornarà un vector amb els valors del potencial als punts xx:
def potencial(nx,xx):
	V=1j*np.zeros(nx)
	for i in range(0,nx):
		x=xx[i]
		if x <= -xe+a:
			V[i]=y1
		elif x >= xe+a:
			V[i]=y1
		else:
			V[i]=b*(m/2.0)*(omega*(x-a))**2
	return V
	
V=potencial(nx,xx)
print(V)

plt.plot(xx,V)
plt.xlabel('x')
plt.ylabel('V(x)')
plt.show()