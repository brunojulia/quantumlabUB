"""
Jan Albert Iglesias 20/03/2018
"""

"This program computes the movement of a ball rolling without slipping on a gaussian-like ground."

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Gaussian function parameters:
sigma = 0.3
mu = 0.

#Ball parameters:
R = 0.1 #m
m = 20. #kg

#Gravity:
g = 9.806 #m/s^2

#Parameters used to solve the differential equation:
steps = 150
finalt = 2.6 #s
deltat = finalt/float(steps)

dx = 0.001 #Approximate step used to compute the perimeter.

#Initial conditions; initial position and velocity of the contact point.
yin = np.array([0.05,0.0])


"The Gaussian funtion and its derivatives."
def fgauss(x):
    """
    Gaussian function.
    """
    f = 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-(x-mu)**2/(2.*sigma**2))
    #f = (x-1)**2   #This is for the parabola.
    return f

def dfgauss(x):
    """
    First derivative of the Gaussian function.
    """
    df = -(x-mu)*fgauss(x)/sigma**2
    #df = 2*(x-1)    #This is for the parabola.
    return df

def d2fgauss(x):
    """
    Second derivative of the Gaussian function.
    """
    d2f = -(1. - (x-mu)**2/sigma**2)*fgauss(x)/sigma**2
    #d2f = 2        #This is for the parabola.
    return d2f

def d3fgauss(x):
    """
    Third derivative of the Gaussian function.
    """
    d3f = -(-2.*(x-mu)/sigma**2 - (1. - (x-mu)**2/sigma**2)*(x-mu)/sigma**2)*fgauss(x)/sigma**2
    #d3f = 0.       #This is for the parabola.
    return d3f



#In order to build the derivatives function of this differential equation (i.e. the function used by RK4), many calculations need to be done.
#Thus, some intermediate functions have been defined for ease of comprehension.

def alpha(x):
    """
    Angle between the line that is normal to curve and the horizontal line at position x.
    """
    a = np.pi/2. + np.arctan(dfgauss(x))
    return a

def dalpha(x):
    """
    First derivative of alpha(x).
    """
    da = 1./(1.+dfgauss(x)**2)*d2fgauss(x)
    return da

def d2alpha(x):
    """
    Second derivative of alpha(x).
    """
    d2a = -2./(1.+dfgauss(x)**2)**2*dfgauss(x)*d2fgauss(x)**2 + 1./(1.+dfgauss(x)**2)*d3fgauss(x)
    return d2a


"Position of the center of mass"
#We chosed the x coordinate of the contact point between the ball and the ground (the Gaussian function) as the generalized coordinate.
#So everything is solved with respect to this coordinate.

def xcm(x):
    """
    X coordinate of the center of mass position with respect to the x coordinate of the contact point (between the ball and the ground).
    """
    deltax = R*np.cos(alpha(x))
    xcm = x + deltax
    return xcm

def ycm(x):
    """
    Y coordinate of the center of mass position with respect to the x coordinate of the contact point (between the ball and the ground).
    """
    deltay = R*np.sin(alpha(x))
    ycm = fgauss(x) + deltay
    return ycm

def dxcm(x):
    """
    First derivative of xcm with respect to x.
    """
    dx = 1. - R*np.sin(alpha(x))*dalpha(x)
    return dx

def dycm(x):
    """
    First derivative of ycm with respect to x.
    """
    dy = dfgauss(x) + R*np.cos(alpha(x))*dalpha(x)
    return dy


"Derivatives function"
def frollingball(t, yvec):
    """
    Function to be used by RK4.
    It returns the derivatives with respect to time of the variables in yvec (i.e. yvec[0]=x, yvec[1]=xdot).
    """
    x = yvec[0]
    xdot = yvec[1]
    f1 = xdot
    f2 = (-0.7*m*xdot**2*(-2.*dxcm(x)*R*(np.cos(alpha(x))*dalpha(x)**2 + np.sin(alpha(x))*d2alpha(x)) +
         2.*dycm(x)*(d2fgauss(x) + R*(-np.sin(alpha(x))*dalpha(x)**2 + np.cos(alpha(x))*d2alpha(x))))
         - m*g*(dfgauss(x) + R*np.cos(alpha(x))*dalpha(x))) / (1.4*m*(dxcm(x)**2 + dycm(x)**2))
    ftot = np.array([f1,f2])

    return ftot


"Runge Kutta 4"
def RK4(t, deltat, yin, fun):
    """
    It makes one step 'deltat' using the RK4 method.
    This means that given the array 'yin', which contains the values of the variables at time 't', it returns a new array with the values at time = t+deltat.
    'fun' must be a f(t,y)-like function that returns an array containing the derivatives of each variable in y.
    """
    k1 = fun(t, yin)
    k2 = fun(t + deltat/2., yin + (deltat/2.)*k1)
    k3 = fun(t + deltat/2., yin + (deltat/2.)*k2)
    k4 = fun(t + deltat, yin + deltat*k3)
    yout = yin + deltat/6.*(k1 + 2*k2 + 2*k3 + k4)
    return yout


"Solving the differential equation"
#To solve for the rotated angle, the trapezoidal rule of integration needs to be defined.
def trapezoidal(a,b,dx,fcn):
    """
    It integrates fcn(x) from 'a' to 'b' using the trapezoidal rule.
    It uses a step similar to dx that gives an integer number of steps.
    If the desired step dx is bigger than (b-a), just one interval is used.
    """
    Niter = abs(int((b-a)/dx))
    summ = (fcn(a) + fcn(b))/2.

    if Niter == 0:
        deltax = (b-a)

    elif Niter != 0:
        deltax = (b-a)/float(Niter)
        for i in range(1,Niter,1):
            summ = summ + fcn(a + deltax*i)

    #Note that if b<a, the result of the integral is negative.
    return summ*deltax


def gaussperim(x):
    """
    Function to be integrated to compute the perimeter.
    """
    f = np.sqrt(1 + dfgauss(x)**2)
    return f


#The solution is stored in a supermatrix with 3 columns: time, x and xdot.
#The rotated angle with respect to the vertical direction is stored in an array.
supermatrix = np.zeros(shape=(steps,3))
supermatrix[0,:] = np.array([0., yin[0], yin[1]])

angle = np.zeros(shape=(steps,1))
angle[0] = -np.arctan(dfgauss(yin[0]))
perimeter = 0.

for i in range(1,steps,1):
    t = i*deltat
    x0 = yin[0]
    yin = RK4(t, deltat, yin, frollingball)
    supermatrix[i,:] = np.array([t, yin[0], yin[1]])

    #Note that perimeter will be negative if the ball rolls to the left side of the gaussian.
    perimeter = perimeter + trapezoidal(x0, yin[0], dx, gaussperim)
    theta = perimeter/R
    beta = np.arctan(dfgauss(yin[0]))
    angle[i] = theta - beta


"Creating the animation"
fig, ax = plt.subplots()
ln, = plt.plot([],[],'ro', ms=1 ,animated=True)

def init():
    rangex = np.arange(-5,5,0.01)
    ax.plot(rangex,fgauss(rangex), 'g-')
    #ax.plot(xcm(rangex),ycm(rangex))
    ax.axis('scaled') #A squared graph is needed in order to correctly see the dimensions.
    plt.xlim((0,2))
    plt.ylim((0,1.5))
    return ln,

#The used frames are the values of the row index.
def update(row):
    x = supermatrix[row,1]
    XXcm = xcm(x)
    YYcm = ycm(x)
    gamma = np.arange(0,2*np.pi,0.005/R)
    XXr = XXcm + R*np.sin(gamma)
    YYr = YYcm + R*np.cos(gamma)
    XXp1 = XXcm + R*np.sin(angle[row])/2.
    YYp1 = YYcm + R*np.cos(angle[row])/2.
    XXp2 = XXcm + R*np.sin(angle[row] + np.pi/2.)/2. #Changing the initial angle
    YYp2 = YYcm + R*np.cos(angle[row] + np.pi/2.)/2.
    XXp3 = XXcm + R*np.sin(angle[row] + np.pi)/2.
    YYp3 = YYcm + R*np.cos(angle[row] + np.pi)/2.
    XXp4 = XXcm + R*np.sin(angle[row] - np.pi/2.)/2.
    YYp4 = YYcm + R*np.cos(angle[row] - np.pi/2.)/2.
    xarr = np.array([XXcm,XXp1,XXp2,XXp3,XXp4])
    yarr = np.array([YYcm,YYp1,YYp2,YYp3,YYp4])
    ln.set_data(np.append(xarr, XXr), np.append(yarr, YYr))
    return ln,

ani = FuncAnimation(fig, update, frames=range(steps),
      init_func=init, blit=True, interval=60)

plt.show()
