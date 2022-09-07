#%% 1. Initial parameters
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:14:16 2022

@author: Eric Vidal Marcos
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

import os
import imageio



# program to compute the time
# of execution of any python code
import time
 
# we initialize the variable start
# to store the starting time of
# execution of program
start = time.time()
 

 


#Arbitrary spin to study
s=3     #total spin
dim=round(2*s+1)    #in order to work when s is half-integer with int dim
Nterm1=s*(s+1)      #1st term of N+-

counter=0

filenames = []

#Hamiltonian Parameters
for D in np.linspace(1, 10, 5):
    for B in np.linspace(0.1, 3.1, 10):
        for hz in np.linspace(0.1, 3.1, 50):
            #Time span
            tf=(D/hz)*(2*(s-1))+10
            At=[-10,tf]
            
            #IC
            a_m0=[]
            a_m0.append(1+0j)
            for i in range(dim-1):
                a_m0.append(0+0j)
            
            
            
            
            #Transition times
            #Because of Hamiltonian symetry transitions can only occur s times
            #which correspond to each level from the metastable state opposite to the true
            #ground state, and then goes up or down dependig which is the true ground state
            #m=s or m=-s, and changes from steps of 2
            time_n=[]
            for i in range(s):
                time_n.append((D/hz)*(2*i))
            
            #States energies if H_0
            energies=[]
            for i in range(dim):
                energies.append([])
            for i in range(dim):
                for j in range(2):
                    energies[i].append(-D*(i-s)**2-hz*At[j]*(i-s))
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
                eigenterm=am[k]*(-D*kreal**2-hz*t*kreal)
                
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
            #print(odes(0, a_m0))
            
            #%% 3. Resolution and plotting
            
            #solve
            a_m=solve_ivp(odes, At, a_m0)
            
            #Plotting parameters
            t=a_m.t[:]  #time
            
            aplot=[]
            for i in range(dim):
                aplot.append(np.abs(a_m.y[i,:])**2)     #Probabilities coeff^2
            
            norm = np.sum(aplot, axis=0)    #Norm (sum of probs)
            
            #Plot
            plt.figure()
            #plt.subplot(121)
            plt.title('D={}, $h_z$={:.3f}, B={:.3f}'.format(D, hz, B))
            plt.xlabel('t')
            plt.ylabel('$|a|^2$')
            plt.axhline(y=1.0,linestyle='--',color='grey')
            for i in range(s):
                plt.axvline(x=time_n[i],linestyle='--',color='grey')
            for i in range(dim):
                plt.plot(t, aplot[i],'-',label='m='+str(i-s))
            plt.plot(t, norm,'-',label='norma')
            plt.legend(bbox_to_anchor=(0,0.5), loc='center left')
            #plt.subplot(122)
            #plt.title('States energies if $\mathcal{H}_0$')
            #plt.xlabel('t')
            #plt.ylabel('$E$')
            #for i in range(s):
            #    plt.axvline(x=time_n[i],linestyle='--',color='grey')
            #for i in range(dim):
            #    plt.plot(At, energies[i],'-',label='$E_{'+str(i-s)+'}$')
            #plt.legend()
            
            #plt.tight_layout()
            
            
            # create file name and append it to a list
            filename = f'{hz}.png'
            filenames.append(filename)
            
            # save frame
            plt.savefig(filename)
            plt.close()# build gif
        with imageio.get_writer('D{:.3f}B{:.3f}hzs.gif'.format(D, B), mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                
        # Remove files
        for filename in set(filenames):
            os.remove(filename)
        counter+=1
        print('GIF num: {}'.format(counter))
        



# now we have initialized the variable
# end to store the ending time after
# execution of program
end = time.time()
 
# difference of start and end variables
# gives the time of execution of the
# program in between
print("The time of execution of above program is :", end-start)