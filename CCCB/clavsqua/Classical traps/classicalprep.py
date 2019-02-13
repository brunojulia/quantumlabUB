"""
Jan Albert Iglesias
11/02/2019

This program pre-computes the evolution of the classical system.
The desired directory needs to be created and specified. It'll write 3 fiiles; one for the supermatrix_cla
(time, position and velocity), one for the angles and one for the initial (though it's constant) energy
(it is actually the height equivalent to the total energy, as if it were all potential energy).
"""

import rollball as rob
import numpy as np

"Definitions"
#File:
dir = "Demo3_cla"
fsuper = dir + "/super.npy"
fang   = dir + "/ang.npy"
fene   = dir + "/ene.npy"

#Ball:
R = 0.2
rob.m = 5

#Physical constants:
rob.g = 9.806

#Potential:
height = 1
sigma_cla = 1./(np.sqrt(2*np.pi)*height)
mu_cla = 0
k_cla = 0.8

#Initial conditions:
x0 = -1.5
xdot0 = 0.0
yin0 = np.array([x0, xdot0])
time_cla = 0.
dtime0_cla = 0.01 #Default time step (used by RK4 and as default playing velocity).

deltax_cla = 0.005 #dx to integrate the ground perimeter.

#Clock (Classical):
time_cla = 0.
dtime0_cla = 0.01 #Default time step (used by RK4 and as default playing velocity).

#Runge-Kutta-Fehlberg:
rob.eps = 0.000001 #Tolerance
rob.h = 1
yarr = np.zeros(shape=(2,2))
tvec = np.zeros(shape=(2,1))


"Computing"
#Initiallization:
yin = yin0
rob.h = 1    #Initial RKF45 step.
yarr[1,:] = yin0
tvec[1] = 0
lastt = dtime0_cla
tmax_cla = 10

#Appending first values.
supermatrix_cla = np.array([[time_cla, yin0[0], yin0[1]]]) #3 columns: time, x and xdot
angle = np.array([[-np.arctan(rob.dfground(mu_cla, sigma_cla, k_cla, yin0[0]))]])
perimeter = 0.

#Computes the initial energy. translational + rotational + potential.
trans = 0.5*rob.m*((rob.dxcm(R, mu_cla, sigma_cla, k_cla, yin0[0])*yin0[1])**2
+ (rob.dycm(R, mu_cla, sigma_cla, k_cla, yin0[0])*yin0[1])**2)
rot = 0.2*rob.m*R**2*(rob.groundperim(mu_cla, sigma_cla, k_cla, yin0[0])/R
- rob.dalpha(mu_cla, sigma_cla, k_cla, yin0[0]))**2*yin0[1]**2
pot = rob.m*rob.g*rob.ycm(R, mu_cla, sigma_cla, k_cla, yin0[0])
energy = trans + rot + pot
Eheight = energy/(rob.m*rob.g) - R  #Height equivalent to the total energy. (As if it were all potential energy).

#Defining:
def extend():
    """
    It extends the matrices with the evolution parameters by the RKF45 method.
    """
    global lastt, yin, supermatrix_cla, perimeter, angle
    #Makes one step by RKF.
    yarr[0,:] = yarr[1,:]
    yarr[1,:] = rob.RKF(R, mu_cla, sigma_cla, k_cla, tvec[1], yarr[1,:], rob.frollingball)

    tvec[0] = tvec[1]
    tvec[1] = tvec[1] + rob.h

    #Fills all the possible values between the last step and the new one by interpolation.
    while lastt < tvec[1]:
        x0 = yin[0]

        #Position & velocity:
        yin = rob.interpol(tvec, yarr, lastt)
        supermatrix_cla = np.concatenate((supermatrix_cla, [[lastt, yin[0], yin[1]]]))

        #Angle:
        perimeter = perimeter + rob.trapezoidal(mu_cla, sigma_cla, k_cla, x0, yin[0], deltax_cla, rob.groundperim)
        theta = perimeter/R
        beta = np.arctan(rob.dfground(mu_cla, sigma_cla, k_cla, yin[0]))
        angle = np.concatenate((angle, [[theta - beta]]))

        lastt = lastt + dtime0_cla


#Extending it as many times as needed:
while lastt <= tmax_cla + 2*dtime0_cla:
    extend()

"Saving on a file"
np.save(fsuper, supermatrix_cla)
np.save(fang, angle)
np.save(fene, Eheight)

"Reading the file" #Not needed, just to see how it's read.
r_supermatrix_cla = np.load(fsuper)
r_angle = np.load(fang)
r_Eheight = np.load(fene)
