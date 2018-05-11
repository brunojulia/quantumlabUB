"""
Jan Albert Iglesias 2/05/2018

This program computes the movement of a ball rolling without slipping on a given ground.
"""

import numpy as np



"The Gaussian funtion and its derivatives."
def fgauss(mu, sigma, x):
    """
    Gaussian function.
    """
    f = 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-(x-mu)**2/(2.*sigma**2))
    return f

def dfgauss(mu, sigma, x):
    """
    First derivative of the Gaussian function.
    """
    df = -(x-mu)*fgauss(mu, sigma, x)/sigma**2
    return df

def d2fgauss(mu, sigma, x):
    """
    Second derivative of the Gaussian function.
    """
    d2f = -(1. - (x-mu)**2/sigma**2)*fgauss(mu, sigma, x)/sigma**2
    return d2f

def d3fgauss(mu, sigma, x):
    """
    Third derivative of the Gaussian function.
    """
    d3f = -(-2.*(x-mu)/sigma**2 - (1. - (x-mu)**2/sigma**2)*(x-mu)/sigma**2)*fgauss(mu, sigma, x)/sigma**2
    return d3f


"The harmonic function and its derivatives."
def fharm(k, x):
    """
    Harmonic function.
    """
    f = 0.5*k*x**2
    return f

def dfharm(k, x):
    """
    First derivative of the Harmonic function.
    """
    df = k*x
    return df

def d2fharm(k, x):
    """
    Second derivative of the Harmonic function.
    """
    d2f = k
    return d2f

def d3fharm(k, x):
    """
    Third derivative of the Harmonic function.
    """
    d3f = 0.
    return d3f


"Ground and its derivatives."
def fground(mu, sigma, k, x):
    f = fgauss(mu, sigma, x) + fharm(k, x)
    return f

def dfground(mu, sigma, k, x):
    df = dfgauss(mu, sigma, x) + dfharm(k, x)
    return df

def d2fground(mu, sigma, k, x):
    d2f = d2fgauss(mu, sigma, x) + d2fharm(k, x)
    return d2f

def d3fground(mu, sigma, k, x):
    d3f = d3fgauss(mu, sigma, x) + d3fharm(k, x)
    return d3f


def groundperim(mu, sigma, k, x):
    """
    Function to be integrated to compute the perimeter.
    """
    f = np.sqrt(1 + dfground(mu, sigma, k, x)**2)
    return f




#In order to build the derivatives function of this differential equation (i.e. the function used by RK4), many calculations need to be done.
#Thus, some intermediate functions have been defined for ease of comprehension.

def alpha(mu, sigma, k, x):
    """
    Angle between the line that is normal to curve and the horizontal line at position x.
    """
    a = np.pi/2. + np.arctan(dfground(mu, sigma, k, x))
    return a

def dalpha(mu, sigma, k, x):
    """
    First derivative of alpha(mu, sigma, k, x).
    """
    da = 1./(1.+dfground(mu, sigma, k, x)**2)*d2fground(mu, sigma, k, x)
    return da

def d2alpha(mu, sigma, k, x):
    """
    Second derivative of alpha(mu, sigma, k, x).
    """
    d2a = -2./(1.+dfground(mu, sigma, k, x)**2)**2*dfground(mu, sigma, k, x)*d2fground(mu, sigma, k, x)**2 + 1./(1.+dfground(mu, sigma, k, x)**2)*d3fground(mu, sigma, k, x)
    return d2a


"Position of the center of mass"
#We chosed the x coordinate of the contact point between the ball and the ground (the Gaussian function) as the generalized coordinate.
#So everything is solved with respect to this coordinate.

def xcm(R, mu, sigma, k, x):
    """
    X coordinate of the center of mass position with respect to the x coordinate of the contact point (between the ball and the ground).
    """
    deltax = R*np.cos(alpha(mu, sigma, k, x))
    xcm = x + deltax
    return xcm

def ycm(R, mu, sigma, k, x):
    """
    Y coordinate of the center of mass position with respect to the x coordinate of the contact point (between the ball and the ground).
    """
    deltay = R*np.sin(alpha(mu, sigma, k, x))
    ycm = fground(mu, sigma, k, x) + deltay
    return ycm

def dxcm(R, mu, sigma, k, x):
    """
    First derivative of xcm with respect to x.
    """
    dx = 1. - R*np.sin(alpha(mu, sigma, k, x))*dalpha(mu, sigma, k, x)
    return dx

def dycm(R, mu, sigma, k, x):
    """
    First derivative of ycm with respect to x.
    """
    dy = dfground(mu, sigma, k, x) + R*np.cos(alpha(mu, sigma, k, x))*dalpha(mu, sigma, k, x)
    return dy

def d2xcm(R, mu, sigma, k, x):
    """
    Second derivative of xcm with respect to x.
    """
    d2x = -R*(np.cos(alpha(mu, sigma, k, x))*dalpha(mu, sigma, k, x)**2 + np.sin(alpha(mu, sigma, k, x))*d2alpha(mu, sigma, k, x))
    return d2x

def d2ycm(R, mu, sigma, k, x):
    """
    Second derivative of ycm with respect to x.
    """
    d2y = (d2fground(mu, sigma, k, x) + R*(-np.sin(alpha(mu, sigma, k, x))*dalpha(mu, sigma, k, x)**2 + np.cos(alpha(mu, sigma, k, x))*d2alpha(mu, sigma, k, x)))
    return d2y


"Derivatives function"
def frollingball(R, mu, sigma, k, t, yvec):
    """
    Function to be used by RK4.
    It returns the derivatives with respect to time of the variables in yvec (i.e. yvec[0]=x, yvec[1]=xdot).
    """
    x = yvec[0]
    xdot = yvec[1]
    f1 = xdot
    f2 = (-0.5*m*xdot**2*(2.*dxcm(R, mu, sigma, k, x)*d2xcm(R, mu, sigma, k, x) + 2.*dycm(R, mu, sigma, k, x)*d2ycm(R, mu, sigma, k, x))
       -0.4*m*R**2*xdot**2*(groundperim(mu, sigma, k, x)/R - dalpha(mu, sigma, k, x))*(dfground(mu, sigma, k, x)*d2fground(mu, sigma, k, x)/(R*groundperim(mu, sigma, k, x)) - d2alpha(mu, sigma, k, x))
       - m*g*dycm(R, mu, sigma, k, x))/(m*(dxcm(R, mu, sigma, k, x)**2 + dycm(R, mu, sigma, k, x)**2) + 0.4*m*R**2*(groundperim(mu, sigma, k, x)/R - dalpha(mu, sigma, k, x))**2)
    ftot = np.array([f1,f2])

    return ftot


"Runge Kutta 4"
def RK4(R, mu, sigma, k, t, deltat, yin, fun):
    """
    It makes one step 'deltat' using the RK4 method.
    This means that given the array 'yin', which contains the values of the variables at time 't', it returns a new array with the values at time = t+deltat.
    'fun' must be a f(t,y)-like function that returns an array containing the derivatives of each variable in y.
    """
    k1 = fun(R, mu, sigma, k, t, yin)
    k2 = fun(R, mu, sigma, k, t + deltat/2., yin + (deltat/2.)*k1)
    k3 = fun(R, mu, sigma, k, t + deltat/2., yin + (deltat/2.)*k2)
    k4 = fun(R, mu, sigma, k, t + deltat, yin + deltat*k3)
    yout = yin + deltat/6.*(k1 + 2*k2 + 2*k3 + k4)
    return yout


"Solving the differential equation"
#To solve for the rotated angle, the trapezoidal rule of integration needs to be defined.
def trapezoidal(mu, sigma, k, a, b, dx, fcn):
    """
    It integrates fcn(mu, sigma, k, x) from 'a' to 'b' using the trapezoidal rule.
    It uses a step similar to dx that gives an integer number of steps.
    If the desired step dx is bigger than (b-a), just one interval is used.
    """
    Niter = abs(int((b-a)/dx))
    summ = (fcn(mu, sigma, k, a) + fcn(mu, sigma, k, b))/2.

    if Niter == 0:
        deltax = (b-a)

    elif Niter != 0:
        deltax = (b-a)/float(Niter)
        for i in range(1,Niter,1):
            summ = summ + fcn(mu, sigma, k, a + deltax*i)

    #Note that if b<a, the result of the integral is negative.
    return summ*deltax


"""
#The solution is stored in a supermatrix with 3 columns: time, x and xdot.
#The rotated angle with respect to the vertical direction is stored in an array.
#The total energy is stored in another array. (translational, rotational, potential and total)
supermatrix = np.zeros(shape=(steps,3))
supermatrix[0,:] = np.array([0., yin[0], yin[1]])

angle = np.zeros(shape=(steps,1))
angle[0] = -np.arctan(dfground(mu, sigma, k, yin[0]))
perimeter = 0.

energy = np.zeros(shape=(steps,4))
energy[0,0] = 0.5*m*((dxcm(mu, sigma, k, yin[0])*yin[1])**2 + (dycm(mu, sigma, k, yin[0])*yin[1])**2)
energy[0,1] = 0.2*m*R**2*(groundperim(mu, sigma, k, yin[0])/R - dalpha(mu, sigma, k, yin[0]))**2*yin[1]**2
energy[0,2] = m*g*ycm(mu, sigma, k, yin[0])
energy[0,3] = sum(energy[0,:])

for i in range(1,steps,1):
    t = i*deltat
    x0 = yin[0]
    yin = RK4(mu, sigma, k, t, deltat, yin, frollingball)
    supermatrix[i,:] = np.array([t, yin[0], yin[1]])

    #Note that perimeter will be negative if the ball rolls to the left side of the gaussian.
    perimeter = perimeter + trapezoidal(mu, sigma, k, x0, yin[0], dx, groundperim)
    theta = perimeter/R
    beta = np.arctan(dfground(mu, sigma, k, yin[0]))
    angle[i] = theta - beta


    energy[i,0] = 0.5*m*((dxcm(mu, sigma, k, yin[0])*yin[1])**2 + (dycm(mu, sigma, k, yin[0])*yin[1])**2)
    energy[i,1] = 0.2*m*R**2*(groundperim(mu, sigma, k, yin[0])/R - dalpha(mu, sigma, k, yin[0]))**2*yin[1]**2
    energy[i,2] = m*g*ycm(mu, sigma, k, yin[0])
    energy[i,3] = sum(energy[i,:])
"""
