# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:43:37 2021

@author: llucv
"""

import numpy as np
from numpy.random import seed
from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numba
from scipy import special
from numba import jit

seed(1111111111)
c_speed=2.99792458E8
pi=np.pi
e=np.e
Eh=27.21138
kb=8.617333262E-5
h_planck=4.135667696E-15
m_e=0.51099895E6
r_atom=1.5E-10
V_atom=r_atom**3
hbar=h_planck/(2*pi)




@jit(nopython=True)
def state_energy(Z,n):
    energy = -Eh*Z*Z/(2*n*n)
    return energy


def state_n(energy,Z):
    n=np.int_(np.sqrt(-Eh*Z*Z/(2*energy)))
    return n

def discrete_energy(energy,Z):
    n=np.int_(np.sqrt(-Eh*Z*Z/(2*energy)))
    ene=state_energy(Z,n)
    return ene

def discrete_energy_arr(energy,Z):
    ene=np.where(energy < 0,discrete_energy(energy, Z),0)
    return ene


@jit
def Max_Boltz(ground,energy,temp):
    kT=kb*temp
    Ef=((hbar**2)*(3*(pi**2)/V_atom)**(2/3))/(2*m_e)
    mu=Ef*(1-((pi*kT/Ef)**2)/12)
    mu=0
    E=energy-ground
    FD=1/(np.exp((E-mu)/kT)+1)
    prob=E/(np.exp((E-mu)/kT)+1)
    return prob

@jit
def acc_reb_MB(ground,temp):
    energy=rand(1)*ground
    kT=kb*temp
    Ef=((hbar**2)*(3*(pi**2)/V_atom)**(2/3))/(2*m_e)
    mu=Ef*(1-((pi*kT/Ef)**2)/12)
    mu=0
    e_max=kT*(special.lambertw(np.exp(mu/kT-1))+1)
    b=rand(1)*Max_Boltz(0, e_max, temp)
    while b > Max_Boltz(ground,energy,temp):
        energy=rand(1)*ground
        b=rand(1)*Max_Boltz(0, e_max, temp)
    return energy

@jit 
def acc_reb_MB_vec(ground,temp):
    kT=kb*temp
    shape_g=np.shape(ground)
    energy=np.zeros((shape_g[0],shape_g[1]))
    Max=Max_Boltz(0,kT/2,temp)
    for i in range(shape_g[0]):
        for j in range(shape_g[1]):
            energy[i,j]=rand()*ground[i,j]
            b=rand()*Max
            while b > Max_Boltz(ground[i,j],energy[i,j],temp):
                energy[i,j]=rand()*ground[i,j]
                b=rand()*Max
    return energy

@jit
def ph_frequency(energy):
    frequency=h_planck/energy
    return frequency

@jit
def ph_wavelenght(energy):
    wavelenght=h_planck*c_speed/energy
    return wavelenght

@jit
def planck_law_nu(temp,freq):
    kT=kb*temp
    radiance=(2*h_planck*freq**3/c_speed**2)*1/(np.exp(h_planck*freq/kT)-1)
    return radiance

@jit
def planck_law_lambda(temp,lamb):
    kT=kb*temp
    radiance=((2*h_planck*c_speed**2)/(lamb**5))\
                *1/(np.exp(h_planck*c_speed/(kT*lamb)-1))
    
    return radiance

def radiance_wavelenght(data,name,n_columns,temp):
    max_d=(18000E-6/temp)
    min_d=0
    h=(max_d-min_d)/n_columns
    distribution=np.zeros((n_columns))
    wavelenght=np.zeros((n_columns))
    planck_law=np.zeros((n_columns))
    MB_distr=np.zeros((n_columns))
    for i in range(n_columns):
        d0=min_d+(i)*h
        d1=min_d+(i+1)*h
        wavelenght[i]=d0+h/2
        for j in range(np.shape(data)[0]):
            for k in range(np.shape(data)[1]):
                if data[j,k] < d1:
                    if data[j,k] >= d0:
                        distribution[i]=distribution[i]+1
    
    planck_law=planck_law_lambda(temp,wavelenght)
    energy=distribution*h_planck*c_speed/wavelenght
    MB_distr=Max_Boltz(0,h_planck*c_speed/wavelenght,temp)
    
    max_MB=max(MB_distr)
    max_en=max(energy)
     
    i=np.where(energy==max(energy))
    wien_lambda_exp=wavelenght[i]
    
    j=np.where(planck_law==max(planck_law))
    wien_lambda_dis=wavelenght[j]
    max_rad_dis=planck_law_lambda(temp,wien_lambda_dis)
    
    print(h)
    print(wien_lambda_exp)
    print(wien_lambda_exp*temp)
    print(wien_lambda_dis)
    print(wien_lambda_dis*temp)
    print(2.897771955E-3/temp)

    energy=max_rad_dis*energy/max_en
    MB_distr=max_rad_dis*MB_distr/max_MB
    
    plt.plot(wavelenght,energy)
    plt.plot(wavelenght,planck_law)
    plt.plot(wavelenght,MB_distr)
    plt.savefig(name+"_photon_energ.jpg")
    
def bb_start(Np,Nz,Z,temp):
    #nombre atòmic dels nusos
    wall_Z=randint(Nz,size=(2*Np,3*Np))+1
    wall_Z=wall_Z/Z
    
    #energia fonamental dels nusos
    wall_ground_energy=np.zeros((2*Np,3*Np))
    wall_ground_energy=state_energy(wall_Z,1)
    
    #energia de les parets amb la distribució MB
    wall_energy=acc_reb_MB_vec(wall_ground_energy,temp)
    photon_gas=wall_energy
    
    #estat energètic dels nusos i energia del gas de fotons
    wall_energy=discrete_energy_arr(wall_energy,wall_Z)
    photon_gas=photon_gas-wall_energy
    
    black_body_Egze_p012=np.zeros((2*Np,3*Np,6))
    black_body_Egze_p012[:,:,0]=wall_energy
    black_body_Egze_p012[:,:,1]=wall_ground_energy
    black_body_Egze_p012[:,:,2]=wall_Z
    black_body_Egze_p012[:,:,3]=-wall_energy/wall_ground_energy
    black_body_Egze_p012[:,:,4]=photon_gas
    
    return black_body_Egze_p012

def bb_ev(bb_Egze_p012,temp):
    shape_array_bb=np.shape(bb_Egze_p012)
    #excitacions i desexcitacions tèrmiques
    for pas in range(10*shape_array_bb[0]):
        i=randint(shape_array_bb[0])
        j=randint(shape_array_bb[1])
        
        energ=rand()*bb_Egze_p012[i,j,1]
        delta_e=energ-bb_Egze_p012[i,j,0]
        
        if delta_e <= 0:
        #el fotó que es genera pot escapar el cos o quedar-se dins la cavity
            bb_Egze_p012[i,j,4]=bb_Egze_p012[i,j,4]+\
                            delta_e-discrete_energy_arr(energ, 
                                                      bb_Egze_p012[i,j,2])
            
            bb_Egze_p012[i,j,0]=discrete_energy_arr(energ, bb_Egze_p012[i,j,2])
                
        if delta_e > 0:
            kT=kb*temp
            b=rand(1)*Max_Boltz(0, kT/2, temp)
            if b < Max_Boltz(bb_Egze_p012[i,j,1],energ,temp):
            #el fotó que es genera pot escapar el cos o quedar-se dins la cavity
                bb_Egze_p012[i,j,4]=bb_Egze_p012[i,j,4]+\
                            delta_e-discrete_energy_arr(energ,
                                                      bb_Egze_p012[i,j,2])
            
                bb_Egze_p012[i,j,0]=discrete_energy_arr(energ,
                                                        bb_Egze_p012[i,j,2])
        
    #fotons avancen en l'interior de la cavitat (o son  emesos)
    for i in range(shape_array_bb[0]):
        for j in range(shape_array_bb[1]):
            emission=rand()
            if emission <= 0.75:
                l=randint(shape_array_bb[0])
                k=randint(shape_array_bb[1])
                bb_Egze_p012[l,k,5]=bb_Egze_p012[i,j,4]
                bb_Egze_p012[i,j,4]=0
            
    bb_Egze_p012[:,:,0]=bb_Egze_p012[:,:,0]+bb_Egze_p012[:,:,5]
    bb_Egze_p012[:,:,4]=bb_Egze_p012[:,:,0]
    bb_Egze_p012[:,:,0]=discrete_energy_arr(bb_Egze_p012[:,:,0],
                                            bb_Egze_p012[:,:,2])
    bb_Egze_p012[:,:,4]=bb_Egze_p012[:,:,4]-bb_Egze_p012[:,:,0]
    bb_Egze_p012[:,:,3]=-bb_Egze_p012[:,:,0]/bb_Egze_p012[:,:,1]
    
    return bb_Egze_p012
    
print("hello")  
    
T=3000

A=bb_start(64,1,1,T)

plt.imshow(A[:,:,3],cmap="inferno",vmin=-1,vmax=0)
plt.show()

p_nu_e=A[:,:,4]
p_nu_l=ph_wavelenght(p_nu_e)
radiance_wavelenght(p_nu_l,"hst_bb_0_FD",50,T)

print("estat inicial")
for k in range(1):
    A=bb_ev(A,T)

plt.imshow(A[:,:,3],cmap="inferno",vmin=-1,vmax=0)
plt.show()
p_nu_e=A[:,:,4]
p_nu_l=ph_wavelenght(p_nu_e)
radiance_wavelenght(p_nu_l,"hst_bb_200pas",50,T)





    



