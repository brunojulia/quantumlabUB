import numpy as np
import matplotlib.pyplot as plt

# En aquest programa creem un vector amb els valor de la funció phi0
# a partir dels valors del vector xx

nx=30
lmax=7
lmin=-7
lx=lmax-lmin
dx=lx/(nx-1)
hbarra=1
m=1
r=(hbarra**2)/(2*m*(dx**2))

xx=np.linspace(lmin,lmax,nx)

def phi0(nx,xx):
	phi0=1j*np.zeros(nx)
	for i in range(0,nx):
		phi0[i]=(np.exp(-((xx[i])**2)/8.0))/((8*np.pi)**(1.0/2.0))
	return phi0

phi0=phi0(nx,xx)

print(xx)
print(phi0)

plt.plot(xx,phi0)
plt.xlim(-10,10)

plt.show()






