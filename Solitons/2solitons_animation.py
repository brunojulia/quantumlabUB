# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:03:49 2018

@author: Rosa
"""
import numpy as np
import scipy.sparse
from scipy.sparse import diags
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

def bright(z,t,v1,v2,n,z01,z02):
    "Initial state of a system of two bright solitons of the same mass. Solutions for Vext=0"
    imag= 0.0 + 1j
    arg1=((z-z01)*v1)
    arg2=(t*0.5*(1 - v1**2))
    arg3=(-(z-z02)*v2)
    arg4=(t*0.5*(1 - (v2)**2))
    psi=np.sqrt(n)*(1/np.cosh((z-z01) -v1*t))*np.exp(arg1*imag)*np.exp(arg2*imag)
    psi2=np.sqrt(n)*(1/np.cosh((z-z02) +v2*t))*np.exp(arg3*imag)*np.exp(arg4*imag)    
    return psi + psi2
    
def gaussian(x,t,mu,sigma,w):
    return (np.exp(-(x - mu)**2/(2*sigma**2))*np.exp(-0.5j*w*t))/(np.sqrt(2*np.pi*sigma**2))

    
#interaction term of the GP equation    
def interact(g,n,funct):
    """
    Interaction term of the GP equation g is g/|g|=1 for grey solitons, +1
    for bright solitons, n is the density of the infinity for grey solitons and
    the central density n_0 for bright solitons. funct is an array with 
    the state of the system at a certain time.
    """
    return g*np.real(funct*funct.conjugate())/n

#External potential of the GP
def potential(x,x0):
    """
    External potential, typically it will depend on the positon.
    """
    pot=10*(np.exp(-(x)**2/(2*0.5**2)))/(np.sqrt(2*np.pi*0.5**2))
    imag= 0.0 + 1j
    soliton=np.sqrt(1)*(1/np.cosh((x-x0) -t))
    return (pot*0 + 0.1*x**2*0) + soliton*0
    

#we create a figure window, create a single axis in the figure and then
#create a line object which will be modified in the animation.
#we simply plot an empty line, data will be added later
fig = plt.figure()
ax = plt.axes(autoscale_on=False,xlim=(-10, 10), ylim=(0, 4))
plt.xlabel('$\~z$', fontsize=16)
plt.ylabel('$|\psi(\~z)|^2/n$', fontsize=16)
plt.title('Collision of two bright solitons')
line1, = ax.plot([], [], lw=1.5)
line2, =ax.plot([],[], lw=1)


#define the spacing and time interval
limits=12
dz=0.022 #spacing
Nz=(limits-(-limits))/dz #number of points
z=np.linspace(-limits,limits,Nz) #position vector, from -10 to 10 Nz points
dt=0.01 #time interval
#parameters of the solutions of the solitons

v1=0.5#velocity (goes from 0 to 1)
v2=0.5   
n=10 #density, n_inf for grey solitons, n_0 for bright solitons
z01=-5.0 #initial position
z02=5.0
g=-1 #interaction term, -1 for bright solitons, 1 for grey solitons 0 harmonic
w=1#harmonic oscillator    

#sliders, velocity
v1_slider_ax = fig.add_axes([0.2,0.8, 0.15, .015])
v1_slider = Slider(v1_slider_ax, r'$v_1$', valmin = 0, valmax = 1, valinit = v1)
v1_slider.label.set_size(12)
v2_slider_ax = fig.add_axes([0.7,0.8, 0.15, .015])
v2_slider = Slider(v2_slider_ax, r'$v_2$', valmin = 0, valmax = 1, valinit = v2)
v2_slider.label.set_size(12)

#sliders, position
z01_slider_ax = fig.add_axes([0.2,0.85, 0.15, .015])
z01_slider = Slider(z01_slider_ax, r'$z_1$', valmin = -7, valmax = -1, valinit = z01)
z01_slider.label.set_size(12)
z02_slider_ax = fig.add_axes([0.7,0.85, 0.15, .015])
z02_slider = Slider(z02_slider_ax, r'$z_2$', valmin = 1, valmax = 7, valinit = z02)
z02_slider.label.set_size(12)


#sytem at time t, it has to include all the boundary conditions

func_0=[]
for position in z:
    func_0.append(bright(position,0,v1,v2,n,z01,z02))
func_0=np.asanyarray(func_0) #turns to an ndarray (needed for the tridiag solver)

t=0
dif_norm=0
ev_time= 20
steps=int(ev_time/dt)

r= (1j*dt)/(4*dz**2) #parameter of the method 
print('r', r)
print('evolution time:',ev_time)
V=[]
for position in z:
    V.append(2*r*dz**2*potential(position,0))
V=np.array(V)
#to enable interactive plotting
def cn(state):
    #matrixs for the Crank-Nicholson method
    #first [] the numbers to plug in the diagonals, second [] position of the 
    #diagonals (0 is the main one), shape: matrix's shape
    #we compute the main diagonals of the matrices, which in general will depend 
    #on the position z
    mainA=[1+2*r +2*r*dz**2*interact(g,n,state)] #main diagonal of A matrix (time t+ 1)
    mainB=[1-2*r -2*r*dz**2*interact(g,n,state)] #main diagonal of B matrix (time t)
    mainA= np.array(mainA) + V
    mainB= np.array(mainB) - V
    A=diags([-r,mainA,-r],[-1,0,1], shape=(len(state),len(state)))
    A=scipy.sparse.csr_matrix(A) #turs to sparse csr matrix (needed for the tridiag solver)
    B=diags([r,mainB,r],[-1,0,1], shape=(len(state),len(state)))
    #ndarray b product of B and the system's state at time t
    prod=B.dot(state)
    #solver of a tridiagonal problem
    func_1=linalg.spsolve(A,prod)
    return func_1


def init():
    line1.set_data([], [])
    line2.set_data([],[])
    #its important to return the line object, this tells the aimator which objects on the plot
    #to update after each frame
    return line1, line2
    
def animate(i):
    global func_0
    system=cn(func_0)
    line1.set_data(z,np.real(system*np.conjugate(system))/n)
    line2.set_data(z,np.real(potential(z,0)*np.conjugate(potential(z,0)))/n)
    func_0=system
    return line1, line2
    

def button_start(self):
    a=FuncAnimation(fig, animate, init_func=init, frames=steps,interval=1, repeat=False)
    fig.canvas.draw()
    
def initialize(self):
    global func_0
    v1=v1_slider.val
    v2=v2_slider.val
    z01=z01_slider.val
    z02=z02_slider.val
    func_0=[]
    for position in z:
        func_0.append(bright(position,0,v1,v2,n,z01,z02))
    func_0=np.asanyarray(func_0)
    a=FuncAnimation(fig, animate, init_func=init, frames=steps,interval=1, repeat=False)
    fig.canvas.draw()
 
   
initial=fig.add_axes([0.01, 0.8, 0.06, 0.10])
b_initial=Button(initial, 'Start')
b_initial.on_clicked(initialize)

start = fig.add_axes([0.01, 0.69, 0.06, 0.10])
b_start = Button(start, 'Continue')
b_start.on_clicked(button_start)


plt.show()
