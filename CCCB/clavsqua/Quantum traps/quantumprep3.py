"""
Jan Albert Iglesias
06/02/2019

This program pre-computes the eigenvector basis for a specific potential and writes it on a file.
The desired directory needs to be created and specified. It'll write 3 files; one for the eigenvectors
(altogether with the projections), the eigenvalues and the energy.
This is for the 3rd demo, in which the initial wavefunction is a superposition of the first 2 eigenstates.
"""

import timeev as te
import numpy as np

"Definitions"
#File:
dir = "Demo3_qua"
fvecs = dir + "/vecs.npy"
fvals = dir + "/vals.npy"
fene = dir + "/ene.npy"

#Box:
a = -20.
b = 20.
N = 1000
deltax = (b - a)/float(N)
xarr_qua = np.arange(a, b + deltax*0.1, deltax)

#Physical constants:
te.hbar = 4.136 #eV·fs (femtosecond)
m_qua = 1/(2*3.80995) #The value contains hbar^2.

#Potential:
te.factor = 10 #Factor for the applied potential.
height = 13
sigma_qua = 2*te.factor/(np.sqrt(2*np.pi)*height)
mu_qua = 0
k_qua = 0.2

#Method definitions:
Nbasis = 200
coefs = np.zeros(shape = (Nbasis + 1, 1), dtype=complex)
coef_x_efuns = np.zeros(shape = (N + 1, Nbasis + 1), dtype=complex)
evalsbasis = np.zeros(shape = (Nbasis + 1, 1))

"Solving Schrödinger equation"
#Computes the eigenvalues and eigenvectors.
evals, efuns = te.srindwall(a, b, N, m_qua, te.pot, mu_qua, sigma_qua, k_qua)

#Initial wavefunction, superposition.
psivec = 1/np.sqrt(2)*(efuns[:,0] + efuns[:,1])

#Computes the projection of the given wavefunction onto the wavefunction basis.
for j in range(Nbasis + 1):
    prod = np.conjugate(psivec)*efuns[:,j]                       #e*(x)·psi(x)
    coefs[j] = deltax*(np.sum(prod) - prod[0]/2. - prod[-1]/2.)    #Trapezoidal rule for integration.
    coef_x_efuns[:,j] = coefs[j]*efuns[:,j]
    evalsbasis[j,0] = evals[j]                                       #It only uses Nbasis values.

#Computes the energy.
energy = np.sum(np.abs(coefs)**2*evalsbasis)

"Saving on a file"
np.save(fvecs, coef_x_efuns)
np.save(fvals, evalsbasis)
np.save(fene, energy)

"Reading the file" #Not needed, just to see how it's read.
r_coef_x_efuns = np.load(fvecs)
r_evalsbasis = np.load(fvals)
r_energy = np.load(fene)
