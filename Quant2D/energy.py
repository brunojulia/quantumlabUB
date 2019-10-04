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
            Ecin[i,j]=(-1/2.)*(psi[i+1,j]+psi[i-1,j]+psi[i,j+1]+psi[i,j-1]-4.*psi[i,j])/pas**2
    
    #Energy expected
    #psii = np.conjugate(np.transpose(psi))
    psii = np.conjugate(psi)
    Ep = 0.
    Ec = 0.
    
    for i in range (0,n):
        for j in range (0,n):
            Ep = Ep + psii[i,j]*Epot[i,j]
            Ec = Ec + psii[i,j]*Ecin[i,j]
            
    Ep = Ep*pas**2
    Ec = Ec*pas**2
    
    return Ep,Ec,(Ep+Ec)


def Norm(f):
    " Calculate norm where f is wave matrix "
    norm=0.
    n=len(f[0,:])
    pas=0.05
    for i in range (0,n):
        for j in range (0,n):
            norm=norm+abs(np.real(np.conjugate(f[i,j])*f[i,j]))
    return norm*pas**2


def Evaluate1(wgreatest,pts):
    
    N=100
    dx=0.05
    L = N*dx
    
    xx,yy = np.meshgrid(np.arange(-L/2,L/2,dx),np.arange(-L/2,L/2,dx),sparse=True)
    
    ''' Energy of initial and final state of harmonic oscilatior ground eigenstate 
        centered and with different frequences '''
    
    x0 = 0.
    y0 = 0.
    
    w_list = np.zeros(pts)
    Ep_ini = np.zeros(pts)
    Ep_fin = np.zeros(pts)
    Ec_ini = np.zeros(pts)
    Ec_fin = np.zeros(pts)
    E_ini = np.zeros(pts)
    E_fin = np.zeros(pts)

    for i in range (0,pts):
        w=i*wgreatest/float(pts)
        
            #Potential and initial wavefunction
        po = potentials.osc((xx,yy),(x0,y0,w))
        wa = wavef.InitWavef.OsciEigen((xx,yy),(x0,y0,w,0,0))
            #Time evolution
        PSI = wavef.Wave(po,wa)
        PSIEVOL = PSI.CrankEvolution()
            #Initial and final energy
        Ep, Ec, Et = Energy(po, wa)
        Epfin, Ecfin, Etfin = Energy (po, PSIEVOL[99,:,:])
        
        w_list[i] = w
        Ep_ini[i] = float(Ep)
        Ep_fin[i] = float(Epfin)
        Ec_ini[i] = float(Ec)
        Ec_fin[i] = float(Ecfin)
        E_ini[i] = float(Et)
        E_fin[i] = float(Etfin)

    plt.subplot(221)
    plt.title('Efin-Eini')
    plt.plot(w_list, abs(E_fin-E_ini))
    
    plt.subplot(222)
    plt.title('Ep_fin, Ep_ini vs E=w/2')
    plt.plot(w_list, Ep_fin, 'o',w_list, Ep_ini, '^', w_list, w_list/2.)
    
    plt.subplot(223)
    plt.title('Ep, Ec and Etotal at t=0')
    plt.plot(w_list, Ep_ini, 'o', w_list, Ec_ini, '^', w_list, E_ini, 's')
    
    plt.subplot(224)
    plt.title('Ep and Ec and Etotal at t=T')
    plt.plot(w_list, Ep_fin, 'o', w_list, Ec_fin, '^', w_list, E_ini, 's')
    
    plt.show()
    


def Evaluate2(wgreatest,pts):
    
    N=100
    dx=0.05
    L = N*dx
    
    xx,yy = np.meshgrid(np.arange(-L/2,L/2,dx),np.arange(-L/2,L/2,dx),sparse=True)
    
    ''' Energy of initial and final state of harmonic oscilatior ground eigenstate 
        centered and with different frequences '''
    
    x0 = 0.
    y0 = 0.
    
    w_list = np.zeros(pts)
    Ep_ini = np.zeros(pts)
    Ep_fin = np.zeros(pts)
    Ec_ini = np.zeros(pts)
    Ec_fin = np.zeros(pts)
    E_ini = np.zeros(pts)
    E_fin = np.zeros(pts)
    Norma_ini = np.zeros(pts)
    Norma_fin = np.zeros(pts)

    for i in range (0,pts):
        w=i*wgreatest/float(pts)
        
            #Potential and initial wavefunction
        po = potentials.osc((xx,yy),(x0,y0,w))
        wa = wavef.InitWavef.OsciEigen((xx,yy),(x0,y0,w,0,0))
            #Time evolution
        PSI = wavef.Wave(po,wa)
        PSIEVOL = PSI.CrankEvolution()
            #Initial and final energy
        Ep, Ec, Et = Energy(po, wa)
        Epfin, Ecfin, Etfin = Energy (po, PSIEVOL[99,:,:])
        
        w_list[i] = w
        Ep_ini[i] = float(Ep)
        Ep_fin[i] = float(Epfin)
        Ec_ini[i] = float(Ec)
        Ec_fin[i] = float(Ecfin)
        E_ini[i] = float(Et)
        E_fin[i] = float(Etfin)
        Norma_ini[i] = Norm(wa)
        Norma_fin[i] = Norm(PSIEVOL[99,:,:])

    plt.subplot(221)
    plt.title('|Efin-Eini|')
    plt.plot(w_list, abs(E_fin-E_ini))
    
    plt.subplot(222)
    plt.title('Ep_fin(blue), Ep_ini(orange) vs E=w/2')
    plt.plot(w_list, Ep_fin, 'o',w_list, Ep_ini, '^', w_list, w_list/2.)
    
    plt.subplot(223)
    plt.title('|Norma_fin - Norm_ini|')
    plt.plot(w_list,abs(Norma_fin-Norma_ini), 'o')
    
    plt.subplot(224)
    plt.title('Norm_ fin(blue) and Norm_ini(orange)')
    plt.plot(w_list, Norma_fin, 'o', w_list, Norma_ini, '^')
    plt.axhline(y=1.)
    
    plt.show()

#------------------------------------------------------------------------------

Evaluate2(wgreatest=5.,pts=5)
''' Presentación en el intervalo de frecuencias [0,wgreatest]
    pts=numero de puntos (equiespaciados) representados   '''