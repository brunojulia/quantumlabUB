import numpy as np

# Aquest programa ens dona un vector xx de dimensió nx i longitud lx
# L'amplada de cada interval és dx

nx=20
lmax=2
lmin=-2
lx=lmax-lmin
dx=lx/(nx-1)
hbarra=1
m=1
r=(hbarra**2)/(2*m*(dx**2))

xx=np.linspace(lmin,lmax,nx)

for i in range(0,nx):
	#La posición i=0 corresponde a nx=1
	#La posición i=1 corresponde a nx=2
	# ...
	#La posición i=19 corresponde a nx=20
	print(i)


print(dx)  # Intervalo del vector
print(xx)  # Valor del vetor xx