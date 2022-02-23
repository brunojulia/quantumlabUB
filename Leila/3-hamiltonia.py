import numpy as np
import matplotlib.pyplot as plt


nx=4
lmax=7
lmin=-7
lx=lmax-lmin
dx=lx/(nx-1)
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

ham=hamiltonia(nx,r)

print(r)
print(ham)

