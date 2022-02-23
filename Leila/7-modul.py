import numpy as np
import matplotlib.pyplot as plt

nx=20
dx=10

xx=np.linspace(-7,7,nx)

def phi0(nx,xx):
	phi0=1j*np.zeros(nx)
	for i in range(0,nx):
		phi0[i]=(np.exp(-((xx[i])**2)/2.0))/((2*np.pi)**(1.0/2.0))+0.1j*i
	return phi0


def modul(nx,f):
	phi_abs = np.zeros(nx)
	for i in range(0,nx):
		phi_abs=(abs(f))
	return phi_abs

phi0=phi0(nx,xx)
phi_abs=modul(nx,phi0)

print(phi0)
print(phi_abs)

plt.plot(xx,phi0)
plt.show()
	