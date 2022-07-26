#%% 1. Initial parameters
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:14:16 2022

@author: Eric Vidal Marcos
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#Arbitrary spin to study
s = 4     #total spin
dim=round(2*s+1)    #in order to work when s is half-integer with int dim
Nterm1=s*(s+1)      #1st term of N+-

tf = 70
#Hamiltonian Parameters
D = 7
alpha = 1.7
H0 = (tf/2)*np.abs(alpha)
B = 0.35

#Time span
At = [0,tf]

#IC
a_m0=[]

for i in range(dim-1):
    a_m0.append(0+0j)
a_m0.append(1+0j)


#Transition times
#Because of Hamiltonian symetry transitions can only occur s times
#which correspond to each level from the metastable state opposite to the true
#ground state, and then goes up or down dependig which is the true ground state
#m=s or m=-s, and changes from steps of 2
time_n=[]
for i in range(s):
    time_n.append(-(D/alpha)*(2*i)+H0/alpha)

#States energies if H_0
energies=[]
for i in range(dim):
    energies.append([])
for i in range(dim):
    for j in range(2):
        energies[i].append(-D*(i-s)**2+(H0-alpha*At[j])*(i-s))

#%% 2. Coupled differential equations
#WE HAVE TO TAKE INTO ACCOUNT THAT M DOESNT GO FROM -S TO S
#IT GOES FROM 0 TO DIM-1=2s

#N+ and N- definition
def Np(m):
    m=m-s   #cuz m goes from 0 to 2s
    Nplus=np.sqrt(Nterm1-m*(m+1))
    return Nplus
def Nm(m):
    m=m-s   #cuz m goes from 0 to 2s
    Nminus=np.sqrt(Nterm1-m*(m-1))
    return Nminus

#PER TESTEJAR SI EL METODE ES CORRECTE, COMPARAREM AMB RESULTATS LAURA
#definition of ODE's
def dak1(s, k, t, am):
    '''Inputs: s (int) total spin, k (int) quantum magnetic number,
    t (int or float) time, am (array (dim,1)) coefficients of each state.
    Outputs: dak (complex) time derivative of a_k.
    This function returns each differential equation for coefficients time
    evolution.'''
    #First we define k to the scale we work in QM
    kreal=k-s
    if (kreal>s):
        print('It makes no sense that k>s or k<s, sth went wrong.')
        exit()
        
    #eigenvalues term
    eigenterm=am[k]*(-D*kreal**2+(H0-alpha*t)*kreal)
    
    #summatory term
    sumterm=0
    for m in range(dim):
        #first we apply Kronicker deltas
        if (k==(m+2)):
            sumtermpos=Np(m)*Np(m+1)
        else:
            sumtermpos=0
            
        if (k==(m-2)):
            sumtermneg=Nm(m)*Nm(m-1)
        else:
            sumtermneg=0
            
        #and obtain summatory term along the for
        sumterm += am[m]*(B/2)*(sumtermpos+sumtermneg)
    
    #finally obtaining the result of one differential equation
    dak=-1j*(eigenterm+sumterm)
    return dak

def odes(t, a_m):
    '''Input: t (int or float) time and a_m (1D list) coefficients. D, h, B
    (int or floats) are Hamiltonian parameters that could be omitted because
    they are global variables as s (spin).
    Ouput: system (1D list), this is the coupled differential equation
    system.'''
    system=[]
    for i in range(dim):
        system.append(dak1(s, i, t, a_m))
    
    return system

#test the defined function odes
print(odes(0, a_m0))


#%% 3. Resolution and plotting

#solve
a_m=solve_ivp(odes, At, a_m0)

#Plotting parameters
t=a_m.t[:]  #time

aplot=[]
a_m0inv=[]  #Initial state for inverse path
for i in range(dim):
    prob_i=np.abs(a_m.y[i,:])**2
    aplot.append(prob_i.tolist())      #Probabilities coeff^2
    a_m0inv.append(a_m.y[i,-1])        #IC for inverse case
    
#CHANGE OF STATE
alpha=-alpha
H0=-H0
#At2=[At[1],At[1]+(At[1]-At[0])]

#solve
a_m2=solve_ivp(odes, At, a_m0inv)   #Note At2 and a_m0inv

#Plotting parameters
t2=a_m2.t[:]  #time

t2=t2+At[1]     #This is cuz we're not using the actual values, At instead

#Join values
t=t.tolist()
t2=t2.tolist()
tplot=t+t2

aplot2=[]
aplot_tot=[]
for i in range(dim):
    prob_i=np.abs(a_m2.y[i,:])**2
    aplot2.append(prob_i.tolist())     #Probabilities coeff^2
    aplot_tot.append(aplot[i]+aplot2[i])

norm = np.sum(aplot_tot, axis=0)    #Norm (sum of probs)

#Plot
plt.figure()
plt.subplot(211)
plt.title('General spin method, solve_ivp')
plt.xlabel('t')
plt.ylabel('$|a|^2$')
plt.axhline(y=1.0,linestyle='--',color='grey')
for i in range(s):
    plt.axvline(x=time_n[i],linestyle='--',color='grey')
for i in range(dim):
    plt.plot(tplot, aplot_tot[i],'-',label='m='+str(i-s))
plt.plot(tplot, norm,'-',label='norma')
plt.legend()

#Magnetization
aplot_tot=np.array(aplot_tot)
magne=np.zeros(np.size(aplot_tot[0]))
for i in range(dim):
    magne=magne+aplot_tot[i]*(i-s)
    
    
plt.subplot(212)
plt.title('Cicle')
plt.xlabel('H')
plt.ylabel('M')

tplot=np.array(tplot)
thalf=int(np.size(tplot)/2)

tplot1=tplot[:thalf]
Hplot1=-H0+alpha*tplot1
Hplot1=Hplot1.tolist()

tplot2=tplot[thalf:]
Hplot2=H0-alpha*(tplot2-At[1])  #In order to restart, substracting the t0 for
#the second stage of the experiment
Hplot2=Hplot2.tolist()

Hplot=Hplot1+Hplot2
plt.plot(Hplot, magne,'-')

plt.show()
plt.tight_layout()