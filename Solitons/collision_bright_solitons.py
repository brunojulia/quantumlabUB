# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:37:18 2019

@author: Rosa
"""
import numpy as np
import scipy.sparse
from scipy.sparse import diags
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Slider, Button
import pandas as pd


def bright(z,t,v1,v2,n,z01,z02,k):
    "solution for two bright solitons whith Vext=0"
    imag= 0.0 + 1j
    arg1=((z-z01)*v1)
    arg2=(t*0.5*(1 - v1**2))
    arg3=(-(z-z02)*v2)
    arg4=(t*0.5*(k - v2**2))
    psi=np.sqrt(n)*(1/np.cosh((z-z01) -v1*t))*np.exp(arg1*imag)*np.exp(arg2*imag)
    psi2=np.sqrt(k*n)*(1/np.cosh(np.sqrt(k)*(z-z02 +v2*t)))*np.exp(arg3*imag)*np.exp(arg4*imag)
    return psi  + psi2
    
def gaussian(x,t,mu,sigma,w):
    "gaussian function"
    return (np.exp(-(x - mu)**2/(2*sigma**2))*np.exp(-0.5j*w*t))/(np.sqrt(2*np.pi*sigma**2))

    
def grey(z,t,v,n,z0):
    "solution for a dark soliton whith Vext=0"
    imag=0.0 +1j
    psi=np.sqrt(n)*(imag*v + np.sqrt(1-v**2)*np.tanh((z-z0)/np.sqrt(2)-v*t)*np.sqrt(1-v**2))
    return psi

def d_bright(z,t,v1,n,z01):
    "analytic first derivative for a bright soliton"
    imag= 0.0 + 1j
    dpsi= imag*v1*bright(z,t,v1,0,n,z01,0,0) - np.tanh(z-z01-v1*t)*bright(z,t,v1,0,n,z01,0,0)
    return dpsi
    
#normalitzation function
def Normalitzation(array,h):
    """
    Computes the normalitzation constant using the simpson's method for a 1D integral
    the function to integrate is an array, h is the spacing between points 
    and returns a float
    """
    constant=0.0
    for i in range(len(array)):
        if (i == 0 or i==len(array)):
            constant+=array[i]*array[i].conjugate()
        else:
            if (i % 2) == 0:
                constant += 2.0*array[i]*array[i].conjugate()
            else:
                constant += 4.0*array[i]*array[i].conjugate()
    constant=(h/3.)*constant
    return np.sqrt(1/np.real(constant))
    
#simpsion method
def Simpson(array,h):
    """
    Simpson's method for a 1D integral. Takes the function to integrate as
    an array. h is the spacing between points. Returns a float
    """
    suma=0.0
    for i in range(len(array)):
        if (i==0 or i==len(array)):
            suma+=array[i]
        else:
            if (i % 2) == 0:
                suma+= 2.0*array[i]
            else:
                suma+= 4.0*array[i]
    suma=(h/3.)*suma
    return suma

    
#interaction term of the GP equation    
def interact(g,n,funct):
    """
    Interaction term needed in the CN method where g is g/|g|=1 for grey solitons, +1
    for bright solitons, n is the density of the infinity for grey solitons and
    the central density n_0 for bright solitons. funct is an array with 
    the state of the system at a certain time.
    """
    return g*np.real(funct*funct.conjugate())/n


def potential(x,a):
    """
    External potential, typically it will depend on the positon.
    """
    pot=a*x**2
    return pot 
    
    
def Energies(x,dx,g,n,V,gint):
    """
    Computes the three contributions of the total energy and returns
    the values in a list. It uses the sipmson's method for the integration
    of the expected values. x is an array of the function that we want to 
    compute the energy, dx the spacing between points, g represents the interaction
    g/|g| it will be either -1 for bright solitons or 1 for grey solitons,
    n the central density of the reference soliton, V an array of the external
    potential, gint the value of g
    """
    T=0 #kinetic
    K=0 #interaction
    U=0 #potential
    E=0 #total
    for i in range(1,(len(x)-1),2): #odd
        T+=2*(np.absolute((x[i+1]-x[i-1])/(2*dx)))**2
        K+=2*g*(np.absolute(x[i]))**4/n
        U+=4*(np.real(V[i])*(np.absolute(x[i]))**2)/(n*gint)
    for i in range(2,(len(x)-1),2): #even
        T+=(np.absolute((x[i+1]-x[i-1])/(2*dx)))**2
        K+=g*(np.absolute(x[i]))**4/n
        U+=2*(np.real(V[i])*(np.absolute(x[i]))**2)/(n*gint)
    T+=0.5*(np.absolute((x[1])/(2*dx)))**2 #first point
    T+=0.5*(np.absolute((-x[-2])/(2*dx)))**2 #last point
    T=dx*T/3
    K+=0.5*g*(np.absolute(x[0]))**4/n #first point
    K+=0.5*g*(np.absolute(x[-1]))**4/n #last point
    K=dx*K/3
    U+=(np.real(V[0])*(np.absolute(x[0]))**2)/(n*gint) #first point
    U+=(np.real(V[-1])*(np.absolute(x[-1]))**2)/(n*gint) #last point
    U=dx*U/3
    E=T+K+U
    return[T,K,U,E]
    


"""
TIME AND SPACE DISCRETITZATION
"""
dt=0.005 #time interval
dz=np.sqrt(dt/2) #spacing interval so that fullfills r=0.5 for CN method
limits=12 #z limits, box width 2*limits
Nz=int((limits-(-limits))/dz) #number of z points 
z=np.linspace(-limits,limits,Nz) #position vector, from -limits to limits with Nz points
r= (1j*dt)/(4*dz**2) #parameter of the CN method 
print('r', r)



"""
PLOTING VARIABLES
"""
ev_time= 15 #time units of the plot
interval_frames=0.01 #time interval between frames
steps=int(ev_time/dt) #total number of steps done by cn
frame=int(ev_time/interval_frames) #number of frames
save=int(steps/frame) #interval between cn steps before saving a frame
print('evolution time:',ev_time, 'time units')
print('time steps:',steps)
print('number of frames',frame)

"""
PARAMETERS OF THE SOLITONS
"""
v1=0 #velocity soliton 1 (goes from 0 to 1)
v2=0 #velocity soliton 2 (goes from 0 to 1)
n=10 #density, n_inf for grey solitons, n_0 for bright solitons
z01=-3.5 #initial position soliton 1 (from 8 to -8 if not conflict with the limits of the box)
z02=3.5 #intial position soliton 2 (from 8 to -8 if not conflict with the limits of the box)
k=0 #proportionality between the density of the second soliton
g=-1 #interaction term, -1 for bright solitons, 1 for grey solitons 0 harmonic
w=1#harmonic oscillator
a=0 #potential    


"""
INITIALIZE THE SYSTEM AT t=0
"""
func_0=[] #array of the state of the system at t=0
if g == -1:
    for position in z:
        func_0.append(bright(position,0,v1,v2,n,z01,z02,k))
elif g == 1:
    for position in z:
        func_0.append(grey(position,0,v1,n,0.5))
else:
    for position in z:
        func_0.append(gaussian(position,0,z01,1,w))

func_0=np.asanyarray(func_0) #turns to an ndarray (needed for the tridiag solver)

#we store the norm at t=0 as it will be useful for cheking if it is preserved during
#the time evolution. (not needed for the main simulation, but is an useful chek)
#norm_0=Normalitzation(func_0,dz)
#if g==0:
#    "normalizes to 1 for the gaussian problem"
#    constant=1/np.sqrt(Simpson(np.real(func_0*func_0.conjugate()),dz))
#    func_0=func_0*constant
#    norm_0=Simpson(np.real(func_0*func_0.conjugate()),dz)
#    print('norm_0', norm_0)


"""
CREATE THE WINDOWS OF THE ANIMATION

we create a figure window, create the desired axis in the figure,
and then create a line object which will be modified in the animation.
If we desire a shadowed object we use the fill variable.
For a timer we use the time_template, which will be uptated every iteration.

We also add the sliders which enabel interactive plotting by updating the value
of v1, v2, z01, z02 and k.
"""
fig = plt.figure()

ax1 = plt.axes([0.13, 0.5, 0.8, 0.38]) #solitons
ax2 = plt.axes([0.13,0.14,0.23,0.25]) #phase
ax3 = plt.axes([0.42,0.14,0.23,0.25]) #energy
ax4 = plt.axes([0.7,0.14,0.23,0.25]) #mass

"""
Solitons
"""
ax1.set_xlim(-limits, limits)
ax1.set_ylim(0,6)
ax1.set_xlabel('$\~z$', fontsize=10)
ax1.set_ylabel('$|\psi(\~z)|^2/n$', fontsize=10)
line1a, = ax1.plot(z, np.real(func_0*np.conjugate(func_0))/n, lw=1) #solitons
filling1= ax1.fill_between(z,np.real(func_0*np.conjugate(func_0))/n,y2=0, alpha=0) #soliton's filling


"""
Phase
"""
ax2.set_xlabel('$\~z$',fontsize=10)
ax2.set_ylabel('$angle$ [rad]',fontsize=10)
ax2.set_xlim(-limits, limits)
ax2.set_ylim(-3.5,3.5)
line2, = ax2.plot(z, np.angle(func_0), lw=1) #phase



"""
Energy
"""
ax3.set_xlabel('$\~t$',fontsize=10)
ax3.set_ylabel('$\~E$',fontsize=10)

"""
Mass
"""
ax4.set_xlabel('$\~t$',fontsize=10)
ax4.set_ylabel('mass', fontsize=10)


"""
Sliders
"""

#sliders, velocity
v1_slider_ax = fig.add_axes([0.2,0.9, 0.15, .015])
v1_slider = Slider(v1_slider_ax, r'$v_1$', valmin = 0, valmax = 1, valinit = v1)
v1_slider.label.set_size(12)
v2_slider_ax = fig.add_axes([0.7,0.9, 0.15, .015])
v2_slider = Slider(v2_slider_ax, r'$-v_2$', valmin = 0, valmax = 1, valinit = v2)
v2_slider.label.set_size(12)

#sliders, position
z01_slider_ax = fig.add_axes([0.2,0.95, 0.15, .015])
z01_slider = Slider(z01_slider_ax, r'$z_1$', valmin = -7, valmax = -3, valinit = z01)
z01_slider.label.set_size(12)
z02_slider_ax = fig.add_axes([0.7,0.95, 0.15, .015])
z02_slider = Slider(z02_slider_ax, r'$z_2$', valmin = 3, valmax = 7, valinit = z02)
z02_slider.label.set_size(12)

#slider proporcionality
k_slider_ax= fig.add_axes([0.45,0.95, 0.15, .015])
k_slider = Slider(k_slider_ax, r'$k$', valmin = 0, valmax = 3, valinit = k)
k_slider.label.set_size(12)

#slider potential
a_slider_ax = fig.add_axes([0.45, 0.9, 0.15, .015])
a_slider = Slider(a_slider_ax, r'$potential$', valmin = 0, valmax = 0.5, valinit = a)
a_slider.label.set_size(12)

#slider evolution
time_slider_ax=fig.add_axes([0.13, 0.025, 0.8, 0.03])
time_slider = Slider (time_slider_ax, 'time', valmin = 0, valmax=ev_time, valinit=0)
time_slider.label.set_size(12)


"""
Initialitzation of the external potential. We use the function potential to set an external
potential which deppends on the position plus we add two infinite walls at the limits
of the box with the variable infinit_wall.

We use an extra array V that deppends on the parameter r of the CN method and the spacing dz
which is needed when solving the tridiagonal problem.
"""
infinit_wall=10000000 #limits of the box

Vext=[] #array of the external potential
for position in z:
    Vext.append(potential(position,a))
Vext[0]=infinit_wall
Vext[1]=infinit_wall
Vext[-1]=infinit_wall
Vext[-2]=infinit_wall
ax1.plot(z,Vext, color='g')



V=[] #array of the external potential for the CN method
for position in z:
    V.append(2*r*dz**2*potential(position,a)/(n))
V=np.array(V)
V[0]=infinit_wall
V[1]=infinit_wall
V[-1]=infinit_wall
V[-2]=infinit_wall



"""
Initialitzation of the energies, we create empty arrays to save the values of the 
different energy contributions at every time. We also create a times array.
We also iniatialize two mass arrays, which will store the probability density on the
positive side of the z axes and on the negative side.
"""
#energies
E_kin=[] #kinetic energy
E_pot=[] #potential energy
E_int=[] #interaction energy
E_tot=[] #total energy
times=[] #time array

#mass
mass_p=[]
mass_n=[]



start_time = time.time()

def cn(state):
    """
    We define one step of the cn method.
    matrixs for the Crank-Nicholson method: A, B
    main diagonal of the matrixs: mainA, mainB. 
    diags([numbers to plug in the diagonals], [position of the 
    diagonals (0 is the main one)], shape=(matrix's shape))
    
    It uses the global variables g, n, V and r
    """
    mainA=[1+2*r +2*r*dz**2*interact(g,n,state)] #main diagonal of A matrix (time t+ 1)
    mainB=[1-2*r -2*r*dz**2*interact(g,n,state)] #main diagonal of B matrix (time t)
    mainA= np.array(mainA) + V #add the external potential and saves as an np.array
    mainB= np.array(mainB) - V #add the external potential and saves as an np.array
    A=diags([-r,mainA,-r],[-1,0,1], shape=(len(state),len(state)))
    A=scipy.sparse.csr_matrix(A) #turs to sparse csr matrix (needed for the tridiag solver)
    B=diags([r,mainB,r],[-1,0,1], shape=(len(state),len(state)))
    #ndarray b product of B and the system's state at time t
    prod=B.dot(state)
    #solver of a tridiagonal problem
    func_1=linalg.spsolve(A,prod)
    return func_1 

start_time = time.time()

"""
Compute the evolution of the system. We evolve the system starting a t=0 doing a CN step
every dt. When the number of steps done is multiple of save we store the state of the system
at every point to an evolution matrix, ev_mx, and we compute and save the energy of the system.
To store the phase of the soliton we create a phase evolution matrix ph_mx. We only want to plot
the phase of the system where we have density, so we decide to only store the points which fulfill 
the condition that given a position the density at that point is max - density/max =0.4. where max
corresponds to the second point of higher density of the system.
"""


def evolution(state):
    print("I'm computing!")
    t=0 #time
    ev_mx=[] #stores all the states of the system at different times as a matrix
    ph_mx=[] #stores the phase of the system a different times as a matrix
    ev_mx.append(state)#save the state of the system at time t=0
    phase=[]
    for j in range(len(state)):
        if (np.absolute(state[j])**2) > 0.1: #if the desity is lower than 10^-1 then we do not store the phas value
            phase.append(np.angle(state[j]))
        else:
            phase.append(0.0)
    ph_mx.append(np.angle(phase))
    count=1 #counter, set to 1 as we already have one step which corresponds to t=0
    for i in range(steps):
        system=cn(state)
        if count%save==0:
            ev_mx.append(system)
            #energy
            ene_system=Energies(system,dz,g,n,Vext,1) #avalues the energy
            E_kin.append(ene_system[0])
            E_int.append(ene_system[1])
            E_pot.append(ene_system[2])
            E_tot.append(ene_system[3])
            times.append(t)
            #phase
            phases=[]
            for j in range(len(system)):
                if (np.absolute(system[j])**2) > 0.1:
                    phases.append(np.angle(system[j]))
                else:
                    phases.append(0.0)
            ph_mx.append(phases)
        t+=dt
        count+=1
        state=system
    #ploting the energy
    ax3.plot(times, E_kin, label='E_kin')
    ax3.plot(times, E_int, label='E_int')
    ax3.plot(times, E_pot, label='E_pot')
    ax3.plot(times, E_tot, label='E_tot')
    ax3.legend(loc=9,bbox_to_anchor=(0.5, 1.18), ncol=4) 
    
    for i in range(len(times)):
        positive=[]
        negative=[]
        for j in range(int(len(z)/2)):
            positive.append(np.absolute((ev_mx[i][-j]))**2/n)
            negative.append(np.absolute((ev_mx[i][j]))**2/n)
        mass_p.append(Simpson(positive,dz))
        mass_n.append(Simpson(negative,dz))
    ax4.plot(times,mass_p, label='positive')
    ax4.plot(times,mass_n, label='negative')
    ax4.legend(loc=9,bbox_to_anchor=(0.5, 1.18), ncol=2) 
    print("I'm done!")
    return ev_mx, ph_mx



#"""
#Save to a CSV
#we save to a csv the evolution matrix, the phase evolution matrix, all the time dependent
#variables (energy and mass) and to another the space dependent variables (spacing and potential)
#"""
#print("I'm computing!")
#ev_mx=[]
#ph_mx=[]
#sys_mx=evolution(func_0)
#ev_mx=sys_mx[0]
#ph_mx=sys_mx[1]
#
#print('computing time',time.time()-start_time,'s')
#
#ev_mx_out=pd.DataFrame(ev_mx)
#ev_mx_out.to_csv('ev_2_v0-1_k2.csv', index=None, sep=',', header=False, encoding='utf-8')
#ph_mx_out=pd.DataFrame(ph_mx)
#ph_mx_out.to_csv('ph_2_v0-1_k2.csv', index=None,sep=',', header=False, encoding='utf-8')
#times_out=pd.DataFrame(times)
#ekin_out=pd.DataFrame(E_kin)
#epot_out=pd.DataFrame(E_pot)
#eint_out=pd.DataFrame(E_int)
#etot_out=pd.DataFrame(E_tot)
#mass_p_out=pd.DataFrame(mass_p)
#mass_n_out=pd.DataFrame(mass_n)
#headst=['times', 'ekin', 'epot', 'eint', 'etot', 'mp', 'mn']
#time_out=pd.concat([times_out,ekin_out,epot_out,eint_out, 
#                        etot_out, mass_p_out, mass_n_out], axis=1, sort=False)
#time_out.to_csv('time_2_v0-1_k2.csv', index=None, sep=',', header=headst, encoding='utf-8')
#z_out=pd.DataFrame(z)
#v_out=pd.DataFrame(Vext)
#headse=['z','Vext']
#space_out=pd.concat([z_out,v_out], axis=1, sort=False)
#space_out.to_csv('space_2_v0-1_k2.csv', index=None, sep=',', header=headse, encoding='utf-8')


"""
Read the examples from the files
"""
#initialize the variables, example 1
E_kin1=[] 
E_pot1=[] 
E_int1=[] 
E_tot1=[] 
times1=[] 
z1=[]
mass_p1=[]
mass_n1=[]
Vext1=[]
ev_mx1=[]
ph_mx1=[]
#initialize the variables, example 2
E_kin2=[] 
E_pot2=[] 
E_int2=[] 
E_tot2=[] 
times2=[] 
z2=[]
mass_p2=[]
mass_n2=[]
Vext2=[]
ev_mx2=[]
ph_mx2=[]
#initialize the variables, example 3
E_kin3 =[] 
E_pot3 =[] 
E_int3 =[] 
E_tot3 =[] 
times3 =[] 
z3 =[]
mass_p3 =[]
mass_n3 =[]
Vext3 =[]
ev_mx3 =[]
ph_mx3 =[]
#open the files
in1_ev=pd.read_csv('ev_1_null.csv', header=None)
in1_ph=pd.read_csv('ph_1_null.csv', header=None)
in1_time=pd.read_csv('time_1_null.csv', header=0)
in1_space=pd.read_csv('space_1_null.csv', header=0)
in2_ev=pd.read_csv('ev_2_v1-1_k2.csv', header=None)
in2_ph=pd.read_csv('ph_2_v1-1_k2.csv', header=None)
in2_time=pd.read_csv('time_2_v1-1_k2.csv', header=0)
in2_space=pd.read_csv('space_2_v1-1_k2.csv', header=0)
in3_ev=pd.read_csv('ev_2_v0-1_k2.csv', header=None)
in3_ph=pd.read_csv('ph_2_v0-1_k2.csv', header=None)
in3_time=pd.read_csv('time_2_v0-1_k2.csv', header=0)
in3_space=pd.read_csv('space_2_v0-1_k2.csv', header=0)
#save the files in a list to enable a bucle to extract the values
examples=[[in1_ev,in1_ph,in1_time,in1_space],[in2_ev,in2_ph,in2_time,in2_space],
          [in3_ev,in3_ph,in3_time,in3_space]]
data=[[ev_mx1,ph_mx1,times1,E_kin1, E_pot1,E_int1, E_tot1,mass_p1,mass_n1,z1,Vext1],
      [ev_mx2,ph_mx2,times2,E_kin2, E_pot2,E_int2, E_tot2,mass_p2,mass_n2,z2,Vext2],
      [ev_mx3,ph_mx3,times3,E_kin3, E_pot3,E_int3, E_tot3,mass_p3,mass_n3,z3,Vext3]]

for k in range(3):
    print('file',k,'opened')
    #pick the readed files
    temp_ev=examples[k][0].values
    temp_ph=examples[k][1].values
    temp_time=examples[k][2].values
    temp_space=examples[k][3].values
    #save the evolution matrix and the phase matrix
    for i in range(len(temp_ev)):
        temp2=[]
        temp3=[]
        for j in range (len(temp_ev[i])):
            temp2.append(complex(temp_ev[i][j]))
            temp3.append(float(temp_ph[i][j]))
        data[k][0].append(temp2) #ev_mx
        data[k][1].append(temp3) #ph_mx
    #save the time variables
    for i in range(len(temp_time)):
        data[k][2].append(temp_time[i][0]) #time array
        data[k][3].append(temp_time[i][1]) #e_kin array
        data[k][4].append(temp_time[i][2]) #e_pot array
        data[k][5].append(temp_time[i][3]) #e_int array
        data[k][6].append(temp_time[i][4]) #e_tot array
        data[k][7].append(temp_time[i][5]) #mass_p array
        data[k][8].append(temp_time[i][6]) #mass_n array
    #save the space variables
    for i in range(len(temp_space)):
        data[k][9].append(temp_space[i][0]) #z array
        data[k][10].append(temp_space[i][1]) #Vext array
        
"""
Initialize at the state: example 1
"""
ev_mx=[]
ph_mx=[]
ev_mx=data[0][0]
ph_mx=data[0][1]
#clear the plots of the previous configuration and re-set the labels
ax3.plot(data[0][2], data[0][3], label='E_kin')
ax3.plot(data[0][2], data[0][4], label='E_int')
ax3.plot(data[0][2], data[0][5], label='E_pot')
ax3.plot(data[0][2], data[0][6], label='E_tot')
ax3.legend(loc=9,bbox_to_anchor=(0.5, 1.18), ncol=4)
ax4.plot(data[0][2], data[0][7], label='positive')
ax4.plot(data[0][2], data[0][8], label='negative')
ax4.legend(loc=9,bbox_to_anchor=(0.5, 1.18), ncol=2)    
ax1.plot(data[0][9],data[0][10], color='g')

#add text to show what it is plotted
text1=fig.text(0.01,0.75,'$v_1$=' + str('{0:.2f}'.format(v1)), fontsize=10)
text2=fig.text(0.01,0.7,'$v_2$=' + str('{0:.2f}'.format(v2)), fontsize=10)
text3=fig.text(0.01,0.65,'$z_1$=' + str('{0:.2f}'.format(z01)), fontsize=10)
text4=fig.text(0.01,0.60,'$z_2$=' + str('{0:.2f}'.format(z02)), fontsize=10)
text5=fig.text(0.01,0.55,'k=' + str('{0:.2f}'.format(k)), fontsize=10)
text6=fig.text(0.01,0.5,'a=' + str('{0:.2f}'.format(a)), fontsize=10)



def initialize(self):
    global func_0,ev_mx, ph_mx, E_kin, E_pot, E_tot, E_int, times, Vext, V, counter
    global mass_p, mass_n,text1,text2,text3,text4,text5,text6
    """
    we reset the value of the variables to the ones defined by the sliders
    then we compute again the state of the system
    """
    print("compute_clicked")
    
    #update the values whith the slider's value
    v1=v1_slider.val
    v2=v2_slider.val
    z01=z01_slider.val
    z02=z02_slider.val
    k=k_slider.val
    a=a_slider.val
    
    #clear the plots of the previous configuration and re-set the labels
    ax3.clear()
    ax3.set_xlabel('$\~t$',fontsize=10)
    ax3.set_ylabel('$\~E$',fontsize=10)
    ax4.clear()
    ax4.set_xlabel('$\~t$',fontsize=10)
    ax4.set_ylabel('mass', fontsize=10)
    ax1.clear()
    ax1.set_xlim(-limits, limits)
    ax1.set_ylim(0,6)
    ax1.set_xlabel('$\~z$', fontsize=10)
    ax1.set_ylabel('$|\psi(\~z)|^2/n$', fontsize=10)
    
    #add text to show what it is plotted
    text1.set_visible(False)
    text2.set_visible(False)
    text3.set_visible(False)
    text4.set_visible(False)   
    text5.set_visible(False)
    text6.set_visible(False)
    text1=fig.text(0.01,0.75,'$v_1$=' + str('{0:.2f}'.format(v1)), fontsize=10)
    text2=fig.text(0.01,0.7,'$v_2$=' + str('{0:.2f}'.format(v2)), fontsize=10)
    text3=fig.text(0.01,0.65,'$z_1$=' + str('{0:.2f}'.format(z01)), fontsize=10)
    text4=fig.text(0.01,0.60,'$z_2$=' + str('{0:.2f}'.format(z02)), fontsize=10)
    text5=fig.text(0.01,0.55,'k=' + str('{0:.2f}'.format(k)), fontsize=10)
    text6=fig.text(0.01,0.5,'a=' + str('{0:.2f}'.format(a)), fontsize=10)
    
    #compute again the external potential and the initial state of the system
    Vext=[] #array of the external potential
    for position in z:
        Vext.append(potential(position,a))
    Vext[0]=infinit_wall
    Vext[1]=infinit_wall
    Vext[-1]=infinit_wall
    Vext[-2]=infinit_wall
    ax1.plot(z,Vext, color='g')
    
    V=[] #array of the external potential for the CN method
    for position in z:
        V.append(2*r*dz**2*potential(position,a)/(n))
    V=np.array(V)
    V[0]=infinit_wall
    V[1]=infinit_wall
    V[-1]=infinit_wall
    V[-2]=infinit_wall
    func_0=[]
    for position in z:
        func_0.append(bright(position,0,v1,v2,n,z01,z02,k))
    func_0=np.asanyarray(func_0)
    #energies
    E_kin=[] #kinetic energy
    E_pot=[] #potential energy
    E_int=[] #interaction energy
    E_tot=[] #total energy
    times=[] #time array
    #mass
    mass_p=[]
    mass_n=[]
    sys_mx=evolution(func_0)
    ev_mx=sys_mx[0]
    ph_mx=sys_mx[1]

def ex1(self):
    global ev_mx, ph_mx, E_kin, E_pot, E_tot, E_int, times, mass_p, mass_n
    global text1,text2,text3,text4,text5,text6
    """
    we reset the value of the variables to the ones defined by the sliders
    then we compute again the state of the system
    """
    
    #clear the plots of the previous configuration and re-set the labels
    ax3.clear()
    ax3.set_xlabel('$\~t$',fontsize=10)
    ax3.set_ylabel('$\~E$',fontsize=10)
    ax3.plot(data[0][2], data[0][3], label='E_kin')
    ax3.plot(data[0][2], data[0][4], label='E_int')
    ax3.plot(data[0][2], data[0][5], label='E_pot')
    ax3.plot(data[0][2], data[0][6], label='E_tot')
    ax3.legend(loc=9,bbox_to_anchor=(0.5, 1.18), ncol=4)
    ax4.clear()
    ax4.set_xlabel('$\~t$',fontsize=10)
    ax4.set_ylabel('mass', fontsize=10)
    ax4.plot(data[0][2], data[0][7], label='positive')
    ax4.plot(data[0][2], data[0][8], label='negative')
    ax4.legend(loc=9,bbox_to_anchor=(0.5, 1.18), ncol=2)    
    ax1.clear()
    ax1.set_xlim(data[0][9][0], -data[0][9][0])
    ax1.set_ylim(0,6)
    ax1.set_xlabel('$\~z$', fontsize=10)
    ax1.set_ylabel('$|\psi(\~z)|^2/n$', fontsize=10)
    ax1.plot(data[0][9],data[0][10], color='g')
    
    #add text to show what it is plotted
    text1.set_visible(False)
    text2.set_visible(False)
    text3.set_visible(False)
    text4.set_visible(False)   
    text5.set_visible(False)
    text6.set_visible(False)
    text1=fig.text(0.01,0.75,'$v_1$=' + str(0), fontsize=10)
    text2=fig.text(0.01,0.7,'$v_2$=' + str(0), fontsize=10)
    text3=fig.text(0.01,0.65,'$z_1$=' + str(-5), fontsize=10)
    text4=fig.text(0.01,0.60,'$z_2$=' + str(3.5), fontsize=10)
    text5=fig.text(0.01,0.55,'k=' + str(0), fontsize=10)
    text6=fig.text(0.01,0.5,'a=' + str(0), fontsize=10)

    E_kin=[] #kinetic energy
    E_pot=[] #potential energy
    E_int=[] #interaction energy
    E_tot=[] #total energy
    times=[] #time array
    #mass
    mass_p=[]
    mass_n=[]

    #set the evolution matrix to the saved one
    ev_mx=data[0][0]
    ph_mx=data[0][1]
    
def ex2(self):
    global ev_mx, ph_mx, E_kin, E_pot, E_tot, E_int, times, mass_p, mass_n
    global text1,text2,text3,text4,text5,text6
    """
    we reset the value of the variables to the ones defined by the sliders
    then we compute again the state of the system
    """
    
    #clear the plots of the previous configuration and re-set the labels
    ax3.clear()
    ax3.set_xlabel('$\~t$',fontsize=10)
    ax3.set_ylabel('$\~E$',fontsize=10)
    ax3.plot(data[1][2], data[1][3], label='E_kin')
    ax3.plot(data[1][2], data[1][4], label='E_int')
    ax3.plot(data[1][2], data[1][5], label='E_pot')
    ax3.plot(data[1][2], data[1][6], label='E_tot')
    ax3.legend(loc=9,bbox_to_anchor=(0.5, 1.18), ncol=4)
    ax4.clear()
    ax4.set_xlabel('$\~t$',fontsize=10)
    ax4.set_ylabel('mass', fontsize=10)
    ax4.plot(data[1][2], data[1][7], label='positive')
    ax4.plot(data[1][2], data[1][8], label='negative')  
    ax4.legend(loc=9,bbox_to_anchor=(0.5, 1.18), ncol=2)
    ax1.clear()
    ax1.set_xlim(data[1][9][0], -data[1][9][0])
    ax1.set_ylim(0,6)
    ax1.set_xlabel('$\~z$', fontsize=10)
    ax1.set_ylabel('$|\psi(\~z)|^2/n$', fontsize=10)
    ax1.plot(data[1][9],data[1][10], color='g')

        
    #add text to show what it is plotted
    text1.set_visible(False)
    text2.set_visible(False)
    text3.set_visible(False)
    text4.set_visible(False)   
    text5.set_visible(False)
    text6.set_visible(False)
    text1=fig.text(0.01,0.75,'$v_1$=' + str(1), fontsize=10)
    text2=fig.text(0.01,0.7,'$v_2$=' + str(1), fontsize=10)
    text3=fig.text(0.01,0.65,'$z_1$=' + str(-3.5), fontsize=10)
    text4=fig.text(0.01,0.60,'$z_2$=' + str(3.5), fontsize=10)
    text5=fig.text(0.01,0.55,'k=' + str(2), fontsize=10)
    text6=fig.text(0.01,0.5,'a=' + str(0), fontsize=10)
    
    E_kin=[] #kinetic energy
    E_pot=[] #potential energy
    E_int=[] #interaction energy
    E_tot=[] #total energy
    times=[] #time array
    #mass
    mass_p=[]
    mass_n=[]
    
    #set the evolution matrix to the saved one
    ev_mx=data[1][0]
    ph_mx=data[1][1]
    
def ex3(self):
    global ev_mx, ph_mx, E_kin, E_pot, E_tot, E_int, times, mass_p, mass_n
    global text1,text2,text3,text4,text5,text6
    """
    we reset the value of the variables to the ones defined by the sliders
    then we compute again the state of the system
    """
    
    #clear the plots of the previous configuration and re-set the labels
    ax3.clear()
    ax3.set_xlabel('$\~t$',fontsize=10)
    ax3.set_ylabel('$\~E$',fontsize=10)
    ax3.plot(data[2][2], data[2][3], label='E_kin')
    ax3.plot(data[2][2], data[2][4], label='E_int')
    ax3.plot(data[2][2], data[2][5], label='E_pot')
    ax3.plot(data[2][2], data[2][6], label='E_tot')
    ax3.legend(loc=9,bbox_to_anchor=(0.5, 1.18), ncol=4)
    ax4.clear()
    ax4.set_xlabel('$\~t$',fontsize=10)
    ax4.set_ylabel('mass', fontsize=10)
    ax4.plot(data[2][2], data[2][7], label='positive')
    ax4.plot(data[2][2], data[2][8], label='negative')
    ax4.legend(loc=9,bbox_to_anchor=(0.5, 1.18), ncol=2)
    ax1.clear()
    ax1.set_xlim(data[2][9][0], - data[2][9][0])
    ax1.set_ylim(0,6)
    ax1.set_xlabel('$\~z$', fontsize=10)
    ax1.set_ylabel('$|\psi(\~z)|^2/n$', fontsize=10)
    ax1.plot(data[2][9],data[2][10], color='g')
    
    #add text to show what it is plotted
    text1.set_visible(False)
    text2.set_visible(False)
    text3.set_visible(False)
    text4.set_visible(False)   
    text5.set_visible(False)
    text6.set_visible(False)
    text1=fig.text(0.01,0.75,'$v_1$=' + str(0), fontsize=10)
    text2=fig.text(0.01,0.7,'$v_2$=' + str(1), fontsize=10)
    text3=fig.text(0.01,0.65,'$z_1$=' + str(-3.5), fontsize=10)
    text4=fig.text(0.01,0.60,'$z_2$=' + str(3.5), fontsize=10)
    text5=fig.text(0.01,0.55,'k=' + str(2), fontsize=10)
    text6=fig.text(0.01,0.5,'a=' + str(0), fontsize=10)
    
        
    E_kin=[] #kinetic energy
    E_pot=[] #potential energy
    E_int=[] #interaction energy
    E_tot=[] #total energy
    times=[] #time array
    #mass
    mass_p=[]
    mass_n=[]
    
    #set the evolution matrix to the saved one
    ev_mx=data[2][0]
    ph_mx=data[2][1]

def update(time):
    """
    Updates the plots with the time slider, variable time
    """
    global filling1 
    index= int(time/interval_frames) #evolution index
    filling1.remove()
    line1a.set_ydata(np.absolute(ev_mx[index])**2/n) #solitons
    filling1= ax1.fill_between(z,np.absolute(ev_mx[index])**2/n,y2=0, alpha=0.5, color='blue') #soliton filling
    line2.set_ydata(ph_mx[index]) #phase
    fig.canvas.draw_idle()


time_slider.on_changed(update)
    
"""
Button compute
"""
initial=fig.add_axes([0.01, 0.8, 0.06, 0.10])
b_initial=Button(initial, 'Compute')
b_initial.on_clicked(initialize)

example1=fig.add_axes([0.01, 0.34, 0.06, 0.10])
b_example1=Button(example1, 'Example1')
b_example1.on_clicked(ex1)

example2=fig.add_axes([0.01, 0.22, 0.06, 0.10])
b_example2=Button(example2, 'Example2')
b_example2.on_clicked(ex2)

example3=fig.add_axes([0.01, 0.1, 0.06, 0.10])
b_example3=Button(example3, 'Example3')
b_example3.on_clicked(ex3)



plt.show()



