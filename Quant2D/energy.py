import numpy as np
import matplotlib.pyplot as plt

import wavef
import potentials


def Energy(pot, psi):
    ''' Calculate potential, kinetic and total energy given the potential matrix
        and the wave function matrix'''
        
    n = len(psi[0,:])
    pas=0.05
    Epot = np.zeros((n,n),dtype=complex)
    Ecin = np.zeros((n,n),dtype=complex)
    
    #Energy matrices * |psi>
    Epot = pot*psi
    
    for i in range (1,n-1):
        for j in range (1,n-1):
            Ecin[i,j]=(-1/2.)*(psi[i+1,j]+psi[i-1,j]+psi[i,j+1]+psi[i+1,j-1]-4.*psi[i,j])/pas**2
    
    #Energy expected
    #psii = np.conjugate(np.transpose(psi))
    psii = np.conjugate(psi)
    Ep = 0.
    Ec = 0.
    
    for i in range (0,n):
        for j in range (0,n):
            Ep = psii[i,j]*Epot[i,j]
            Ec = psii[i,j]*Ecin[i,j]
            
    Ep = Ep*pas**2
    Ec = Ec*pas**2
    
    return Ep,Ec,(Ep+Ec)

#------------------------------------------------------------------------

N=100
dx=0.05
L = N*dx

xx,yy = np.meshgrid(np.arange(-L/2,L/2,dx),np.arange(-L/2,L/2,dx),sparse=True)

''' Energy of initial and final state of harmonic oscilatior ground eigenstate 
    centered and with frequence w = 1.5 '''

x0 = 0.
y0 = 0.

pts=5

w_list = np.zeros(pts)
Ep_ini = np.zeros(pts)
Ep_fin = np.zeros(pts)
Ec_ini = np.zeros(pts)
Ec_fin = np.zeros(pts)
E_ini = np.zeros(pts)
E_fin = np.zeros(pts)

#file = open("Energies.dat", "w") 
#file.write('w\tEp_ini\tEc_ini\tEt_ini\tEp_fin\tEc_fin\tEt_fi\n')
for i in range (0,pts):
    w=i*int(10/float(pts))
    
        #Potential and initial wavefunction
    po = potentials.osc((xx,yy),(x0,y0,w))
    wa = wavef.InitWavef.OsciEigen((xx,yy),(x0,y0,w,0,0))
        #Time evolution
    PSI = wavef.Wave(po,wa)
    PSIEVOL = PSI.CrankEvolution()
        #Initial and final energy
    Ep, Ec, Et = Energy(po, wa)
    Epfin, Ecfin, Etfin = Energy (po, PSIEVOL[99,:,:])
    
#    file.write(str(w)+'\t'+str(Ep)+'\t'+str(Ec)+'\t'+str(Et)+'\t'+str(Epfin)+'\t'+str(Ecfin)+'\t'+str(Etfin)+'\n')
    
    w_list[i] = w
    Ep_ini[i] = float(Ep)
    Ep_fin[i] = float(Epfin)
    Ec_ini[i] = float(Ec)
    Ec_fin[i] = float(Ecfin)
    E_ini[i] = float(Et)
    E_fin[i] = float(Etfin)

    
#file.close()

#plt.plot(w_list, Ep_list, 'o', w_list, w_list/2.)
plt.subplot(221)
plt.title('Efin-Eini')
plt.plot(w_list, abs(E_fin-E_ini))

plt.subplot(222)
plt.title('Efin, Eini vs E=w/2')
plt.plot(w_list, E_fin, 'o',w_list, E_ini, '^', w_list, w_list/2.) #E(theoretical)=w/2 for ground state

plt.subplot(223)
plt.title('Ep and Ec at t=0')
plt.plot(w_list, Ep_ini, 'o', w_list, Ec_ini, '^')

plt.subplot(224)
plt.title('Ep and Ec at t=T')
plt.plot(w_list, Ep_fin, 'o', w_list, Ec_fin, '^')

plt.show()

print(E_fin)