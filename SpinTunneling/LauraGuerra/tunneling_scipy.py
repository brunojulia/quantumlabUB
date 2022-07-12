#from sys import settrace
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#FUNCIONS
def H(t):
    Ham = -Sz2 + a * t * Sz + alpha * (Sxy2)
    return Ham

def dam(t,y):
    dadt=[]
    for i in range(mlt):
        sum_H = 0.
        mi = m[i]

        for k in range(mlt):
            mk = m[k]
            mk = np.transpose(mk)
            sum_H = sum_H + y[k] * np.dot(mi, (np.dot(H(t), mk)))

        dam_res = -1j * sum_H
        dadt.append(dam_res)
        dam_res=0

    return dadt

#SELECCIO VALOR DE SPIN
s=float(input("Valor de l'spin:"))
mlt=int(2*s+1)

# Definicio de parametres generals per descriure l,Hamiltonia
Sx=np.zeros(shape=(mlt,mlt),dtype=complex)
Sy=np.zeros(shape=(mlt,mlt),dtype=complex)
Sz=np.zeros(shape=(mlt,mlt),dtype=complex)
m=np.zeros(shape=(mlt,mlt),dtype=int)

for i in range(mlt):
    Sz[i, i] = (i - s)*(-1)
    m[i,i] = 1
    for k in range(mlt):
        if k == (i + 1):
            Sx[i, k] = 0.5 * np.sqrt(s * (s + 1) - (i - s) * (k - s))
            Sy[i, k] = -0.5j * np.sqrt(s * (s + 1) - (i - s) * (k - s))

        if k == (i - 1):
            Sx[i, k] = 0.5 * np.sqrt(s * (s + 1) - (i - s) * (k - s))
            Sy[i, k] = 0.5j * np.sqrt(s * (s + 1) - (i - s) * (k - s))

Sxy2 = np.dot(Sx, Sx) - np.dot(Sy, Sy)
Sz2 = np.dot(Sz, Sz)

y0=[]
for i in range(mlt):
    if i!=(mlt-1):
        y0.append(0+0j)
    else:
        y0.append(1+0j)

alpha = 0.3
a = 0.01
t0 = -20.
tf = 20.
ti=[t0,tf]

sol = solve_ivp(dam, ti, y0)

am=sol.y
mod2=abs(am)*abs(am)

for i in range(mlt-1):
    if i==0:
        norma=(list(map(sum,zip(mod2[i],mod2[i+1]))))
    else:
        norma=(list(map(sum,zip(norma,mod2[i+1]))))
#norma=list(map(sum, zip(mod2)))
t=sol.t

plt.plot(t,norma)

plt.axhline(y=1,xmin=t0,xmax=tf)
for i in range(mlt):
    plt.plot(t,mod2[i])

plt.show()
