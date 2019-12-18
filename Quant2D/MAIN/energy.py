import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
            Ep = Ep + np.real(psii[i,j]*Epot[i,j])
            Ec = Ec + np.real(psii[i,j]*Ecin[i,j])
            
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

def XExpected(psi,order,xx):
    n = len(psi[0,:])
    pas = abs(xx[0,1]-xx[0,0])
    r = order
    j = int(n/2)
    
    XExp = 0.
    psii = np.conjugate(psi)
    
    for i in range(0,n):
        x = xx[0,i]
        XExp = XExp + np.real(psii[j,i]*(x**r)*psi[j,i])
                   
    return XExp*pas

def YExpected(psi,order,yy):
    n = len(psi[0,:])
    pas = abs(yy[1,0]-yy[0,0])
    r = order
    j = int(n/2)
    
    YExp = 0.
    psii = np.conjugate(psi)
    
    for i in range(0,n):
        y = yy[i,0]
        YExp = YExp + np.real(psii[i,j]*(y**r)*psi[i,j])
                   
    return YExp*pas

def XDeviation(psi,xx):
    ''' sigma**2 = <x**2> - <x>**2 '''
    sigma2 = XExpected(psi,2,xx) - (XExpected(psi,1,xx))**2
    
    return np.sqrt(sigma2)

def YDeviation(psi,yy):
    ''' sigma**2 = <x**2> - <x>**2 '''
    sigma2 = YExpected(psi,2,yy) - (YExpected(psi,1,yy))**2
    
    return np.sqrt(sigma2)
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def EvaluateEnergy(wgreatest,pts):
    
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
        
    fig = plt.figure()
    
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('Frequence w')
    ax1.set_title('|Efin-Eini|')
    
    ax1.plot(w_list, abs(E_fin-E_ini), 'o')
    
    
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Frequence w')
    ax2.set_title('Final and initial potential energy and theoretical values')
    
    ax2.plot(w_list, Ep_fin, 'o', label='Final Ep')
    ax2.plot(w_list, Ep_ini, '^', label='Initital Ep')
    ax2.plot(w_list, w_list/2.,label='Theoretical: E=w/2')
    
    legend = ax2.legend(loc='right lower')
    
    
    plt.show()
    


def EvaluateNorm(wgreatest,pts):
    
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
        
    fig = plt.figure()
    
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('Frequence w')
    ax1.set_title('|Norm_fin - Norm_ini|')
    
    ax1.plot(w_list,abs(Norma_fin-Norma_ini), 'o')
    
    
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Frequence w')
    ax2.set_title('Final and initial norm values')
    
    ax2.plot(w_list, Norma_fin,'o',label='Final norm')
    ax2.plot(w_list, Norma_ini,'^', label='Initital norm')
    ax2.axhline(y=1.)
    
    legend = ax2.legend(loc='lower center')
    
    plt.show()
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



def CalcEnForTime():
    file = open('Data-time.txt', 'w')
    xx, yy = np.meshgrid(np.arange(-2.5,2.5,0.05),np.arange(-2.5,2.5,0.05),sparse=True)
    for w in [1.5,2.,3.,4.,5.]:
        print('w=',w)
        file.write('w=' + str(w) + '\n')
        
        initwave = wavef.InitWavef.OsciEigen((xx,yy),(0.,0.,w,0,0))
        pot = potentials.osc((xx,yy),(0,0,w))
        
        norm0 = Norm(initwave)
        Ep0,Ec0,E0 = Energy(pot, initwave)
        
        Epteo = w/2.
        
        df = pd.DataFrame(columns=['dt','T','nor_i','nor_f','Ei','Ef','Epi','Epf','Ept','Eci','Ecf'])
        
        for T in [100,200,400,500,800,1000]:
            print('T=',T)
            dt=1./float(T)
            WAVE = wavef.Wave(pot,initwave,dt,T)
            wavevol = WAVE.CrankEvolution()
            
            normf = Norm(wavevol[T-1,:,:])
            Epf,Ecf,Ef = Energy(pot, wavevol[T-1,:,:])
            
            df = df.append({'dt':dt, 'T':T, 'nor_i':norm0, 'nor_f':normf, 'Ei':E0, 'Ef':Ef, 'Epi':Ep0, 'Epf':Epf ,'Ept':Epteo, 'Eci':Ec0, 'Ecf':Ecf}, ignore_index=True)
                
        file.write(df.to_string())
        file.write('\n')
    
    file.close()

'''
def CalcXExpForTime():
    file = open('Data-Expected.txt', 'w')
    file.write('dt'+'\t'+'T'+'\t'+'<x>'+'\t'+'Sigmax'+'\n')
    xx, yy = np.meshgrid(np.arange(-2.5,2.5,0.05),np.arange(-2.5,2.5,0.05),sparse=True)
    w=3.
    
    for T in [100,200,400,500,800,1000]:
        dt = 1./float(T)
        
        initwave = wavef.InitWavef.OsciEigen((xx,yy),(-1.,0.,w,0,0))
        pot = potentials.osc((xx,yy),(0,0,w))
        
        WAVE = wavef.Wave(pot,initwave,dt,T)
        wavevol = WAVE.CrankEvolution()
        
        xex = XExpected(pot, wavevol[T-1,:,:],1)
        xsig = XDeviation(pot, wavevol[T-1,:,:])
        
        file.write(str(dt)+'\t'+str(T)+'\t'+str(xex)+'\t'+str(xsig)+'\n')
    file.close()
'''
#---------------------------------------------------------------------------

#CalcEnForTime()