import numpy as np
import matplotlib.pyplot as plt

nx=20
lmax=5
lmin=-5
lx=lmax-lmin
dx=lx/(nx-1)

xx=np.linspace(lmin,lmax,nx)

def phi0(nx,xx):
	phi0=1j*np.zeros(nx)
	for i in range(0,nx):
		phi0[i]=(np.exp(-((xx[i])**2)/2.0))/((2*np.pi)**(1.0/2.0))
	return phi0

def modul(nx,f):
	phi_abs = np.zeros(nx)
	for i in range(0,nx):
		phi_abs[i]=abs(f[i])
	return phi_abs

def simpson(h,n,f):

	suma=0.0

	#afegim els extrems
	suma=f[0]+f[len(f)-1]
	print(f[0])
	print(f[len(f)-1])

	#paritats
	for i in range(1,n-1):
		print(f[i])
		if (i%2==0):
			suma=suma+2*f[i]
		else:
			suma=suma+4*f[i]

	suma=(h/float(3))*suma

	return suma

phi0=phi0(nx,xx)
phi_abs=modul(nx,phi0)
area=simpson(dx,nx,phi_abs)

#print(phi0)
print(phi_abs)
print(area)

plt.plot(xx,phi_abs)
plt.show()



