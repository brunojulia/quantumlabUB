# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:31:51 2021

@author: Laura Guerra
"""

import numpy as np
from matplotlib import pyplot as plt
import kivy


''' Definicio de les funcions'''

# Hamiltonia depenent del temps amb unitats d'energia entre D
def H(alpha, t):
    if H_type == 1:
        Ham = -Sz2 + a * t * Sz + alpha * (Sxy2)
    if H_type == 2:
        Ham = -Sz2 + a * t * Sz + alpha * (Sxy2)
    return Ham
    # Ham = -Sz2 + a * t * Sz + alpha * (Sxy2)
    # return Ham


# Equacions diferencials de primer ordre associades a les diferents a_m
def dam(t, am, i):
    sum_H = 0.
    m_i = np.array([m[i]])

    for k in range(mlt):
        m_k = np.array([m[k]])
        m_k = np.transpose(m_k)
        sum_H = sum_H + am[k] * np.dot(m_i, (np.dot(H(alpha, t), m_k)))

    dam_res = -1j * sum_H

    return dam_res


# Funcio que utilitza el metode Runge Kutta4 per resoldre les equacions
# diferencials de primer ordre
def RK4(t, h, am):
    for i in range(mlt):
        k0 = h * dam(t, am, i)
        k1 = h * dam(t + 0.5 * h, am + 0.5 * k0, i)
        k2 = h * dam(t + 0.5 * h, am + 0.5 * k1, i)
        k3 = h * dam(t + 5 * h, am + k2, i)
        am[i] = am[i] + (k0 + 2 * k1 + 2 * k2 + k3) / 6.
    return a_m




""" PROGRAMA """

# Definició de l'Hamiltonià
global H_type
#H_type = int(input('Escriu 1'))
H_type = 1

# Creacio dels vectors propis de l'spin
m = []
s = 10
mlt=int(2*s+1)
dam_res = [0] * (mlt)
alpha = 0.1
a = 0.01
t0 = -50.
tf = 50.

# Definicio de parametres generals per descriure l,Hamiltonia


Sx=np.zeros(shape=(mlt,mlt),dtype=complex)
Sy=np.zeros(shape=(mlt,mlt),dtype=complex)
Sz=np.zeros(shape=(mlt,mlt),dtype=complex)

for i in range(mlt):
    m_i = [0] * (int(2 * s) - i) + [1] + [0] * (i)
    m.append(m_i)

    for k in range(mlt):
        if k == (i + 1):
            Sx[i, k] = 0.5 * np.sqrt(s * (s + 1) - (i - s) * (k - s))
            Sy[i, k] = -0.5j * np.sqrt(s * (s + 1) - (i - s) * (k - s))

        if k == (i - 1):
            Sx[i, k] = 0.5 * np.sqrt(s * (s + 1) - (i - s) * (k - s))
            Sy[i, k] = 0.5j * np.sqrt(s * (s + 1) - (i - s) * (k - s))

        if k == i:
            Sz[i, k] = i - s

Sxy2 = np.dot(Sx, Sx) - np.dot(Sy, Sy)
Sz2 = np.dot(Sz, Sz)


# Condicions inicials
a_m = np.zeros(shape=(mlt, 1), dtype=complex)
a_m[0] = 1 + 0j

# Execucio de les funcions

nstep = 100000
h = (tf - t0) / nstep
t = t0

ti = [0] * (nstep + 1)
mod = [0] * (nstep + 1)

ti[0] = t0

prob = np.zeros(shape=(nstep + 1, mlt))


for k in range(mlt):
    prob[0, k] = abs(a_m[k]) * abs(a_m[k])
mod[0] = np.sum(prob[0])

for n in range(nstep):
    print(n)
    an = RK4(t, h, a_m, s)

    for k in range(mlt):
        prob[n+1,k]=abs(an[k])*abs(an[k])
    ti[n + 1] = t
    mod[n + 1] = np.sum(prob[n+1])
    t = t + h
    a_m = an

plt.title('N=' + str(nstep))
plt.xlabel("t'")
plt.ylabel('a^2')
plt.axhline(y=1.0, linestyle='--', color='grey')
for k in range(mlt):
    plt.plot(ti, prob[:,k], '-', label=str(k-s))
plt.plot(ti, mod, '-', label='m=-1')
plt.show()
