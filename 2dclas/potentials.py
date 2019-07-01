

import numpy as np

#############################

def linear(r,param):
    f = param[0]*r[0] + param[1]*r[1]
    return f

def dlinearx(r,param):
    f = param[0]
    return f

def dlineary(r,param):
    f = param[1]
    return f


#############################
    
def osc(r,param):
    x0 = param[0]
    y0 = param[1]
    k = param[2]
    rlim = param[3]
    rval = np.sqrt((r[0]-x0)**2+(r[1]-y0)**2)
    
    if(rval < rlim):
        f = k*rval**2 - 100
    else:
        vlim = k*rlim**2
        f = 0.
#        f = -2*k*((rlim-x0)*np.exp(-np.abs(r[0]-x0)) + (rlim-y0)*np.exp(-np.abs(r[1]-y0)))
    return f

def doscx(r,param):
    x0 = param[0]
    y0 = param[1]
    k = param[2]
    rlim = param[3]
    
#    if(np.sqrt((r[0]-x0)**2+(r[1]-y0)**2) < rlim):
    if(np.abs(r[0]-x0) < rlim):
        f = k*(2*(r[0]-x0))
#    elif((r[0]-x0) > 0):
    else:
        f = 0.
#        f = -k*2*(rlim-x0)*np.exp(-(np.abs(r[0]-rlim)))
#        f = k*2*(x0-rlim)*np.exp(-(np.abs(r[0]-rlim)))
#    else:
#        f = k*2*(rlim+x0)*np.exp(-(np.abs(r[0]+rlim)))
#        f = k*2*(x0+rlim)*np.exp(-(np.abs(r[0]+rlim)))
    return f

def doscy(r,param):
    x0 = param[0]
    y0 = param[1]
    k = param[2]
    rlim = param[3]
    
#    if(np.sqrt((r[0]-x0)**2+(r[1]-y0)**2) < rlim):
    if(np.abs(r[1]-y0) < rlim):
        f = k*(2*(r[1]-y0))
#    elif((r[1]-y0) > 0):
    else:
        f = 0.
#        f = k*2*(rlim-y0)*np.exp(-(np.abs(r[1]-rlim)))
#        f = k*2*(y0-rlim)*np.exp(-(np.abs(r[1]-rlim)))
#    else:
#        f = k*2*(rlim+y0)*np.exp(-(np.abs(r[1]+rlim)))
#        f = k*2*(y0+rlim)*np.exp(-(np.abs(r[1]+rlim)))
    return f

#############################
    
def woodsaxon(r,param):
    x0 = param[0]
    y0 = param[1]
    V0 = param[2]
    Rx = param[3]
    Ry = param[4]
    theta = param[5]*(np.pi/180.)
    
    a = 1.
    
    x = r[0] - x0
    y = r[1] - y0
    
    xr = x*np.cos(theta) - y*np.sin(theta)
    yr = x*np.sin(theta) + y*np.cos(theta)
    
    px = np.sqrt(xr**2)
    py = np.sqrt(yr**2)
    
    f = V0*(1/(1 + np.exp((px-Rx)/a)))*(1/(1 + np.exp((py-Ry)/a)))
    
    return f

def dwoodsaxonx(r,param):
    x0 = param[0]
    y0 = param[1]
    V0 = param[2]
    Rx = param[3]
    Ry = param[4]
    theta = param[5]*(np.pi/180.)
    
    a = 1.
    
    x = r[0] - x0
    y = r[1] - y0
    
    xr = x*np.cos(theta) - y*np.sin(theta)
    yr = x*np.sin(theta) + y*np.cos(theta)
    
    px = np.sqrt(xr**2)
    py = np.sqrt(yr**2)
    
    sign = np.where(x>0,1,-1)
    
    f = -V0*((sign*np.exp((Rx+px)/a))/(a*(np.exp(Rx/a)+np.exp(px/a))**2))*(1/(1 + np.exp((py-Ry)/a)))
    return f

def dwoodsaxony(r,param):
    x0 = param[0]
    y0 = param[1]
    V0 = param[2]
    Rx = param[3]
    Ry = param[4]
    theta = param[5]*(np.pi/180.)
    
    a = 1.
    
    x = r[0] - x0
    y = r[1] - y0
    
    xr = x*np.cos(theta) - y*np.sin(theta)
    yr = x*np.sin(theta) + y*np.cos(theta)
    
    px = np.sqrt(xr**2)
    py = np.sqrt(yr**2)
    
    sign = np.where(y>0,1,-1)
    
    f = -V0*((sign*np.exp((Ry+py)/a))/(a*(np.exp(Ry/a)+np.exp(py/a))**2))*(1/(1 + np.exp((px-Rx)/a)))
    return f
#############################
    
def gauss(r,param):
    x0 = param[0]
    y0 = param[1]
    V0 = param[2]
    sig = param[3]
    
    x = r[0] - x0
    y = r[1] - y0
    
    f = V0*np.exp(-(x**2)/(2*sig**2))*np.exp(-(y**2)/(2*sig**2))
    
    return f

def dgaussx(r,param):
    x0 = param[0]
    y0 = param[1]
    V0 = param[2]
    sig = param[3]
    
    x = r[0] - x0
    y = r[1] - y0
    
    f = -((V0*x)/(sig**2))*np.exp(-(x**2)/(2*sig**2))*np.exp(-(y**2)/(2*sig**2))
    
    return f

def dgaussy(r,param):
    x0 = param[0]
    y0 = param[1]
    V0 = param[2]
    sig = param[3]
    
    x = r[0] - x0
    y = r[1] - y0
    
    f = -((V0*y)/(sig**2))*np.exp(-(x**2)/(2*sig**2))*np.exp(-(y**2)/(2*sig**2))
    
    return f

#############################
    

def walls(r,param):
    L = param[0]/2. - 0.1*(param[0]/2.)
    a = param[1]
    
    x = r[0]
    y = r[1]
    
    if(np.abs(r[0]) > L or np.abs(r[1]) > L):
        f = a
    elif(r[0] >= 0 and r[1] >= 0):
        x = r[0] - L/1.5
        y = r[1] - L/1.5
        f = (a/4.)*(np.tanh(x) + 1.)*(np.tanh(y) + 1.)
    elif(r[0] > 0 and r[1] < 0):
        x = r[0] - L
        y = -r[1] - L
        f = (a/4.)*(np.tanh(x) + 1.)*(np.tanh(y) + 1.)
    elif(r[0] < 0 and r[1] > 0):
        x = -r[0] - L
        y = r[1] - L
        f = (a/4.)*(np.tanh(x) + 1.)*(np.tanh(y) + 1.)
    elif(r[0] <= 0 and r[1] <= 0):
        x = -r[0] - L
        y = -r[1] - L
        f = (a/4.)*(np.tanh(x) + 1.)*(np.tanh(y) + 1.)
    
    return f
    
def dwallsx(r,param):
    L = param[0]/2. - 0.1*(param[0]/2.)
    a = param[1]
    
    
    if(np.abs(r[0]) > L or np.abs(r[1]) > L):
        f = 0.
    elif(r[0] >= 0 and r[1] >= 0):
        x = r[0] - L/1.5
        y = r[1] - L/1.5
        f = (a/4.)*((1./np.cosh(x))**2)*(np.tanh(y) + 1.)
    elif(r[0] > 0 and r[1] < 0):
        x = r[0] - L
        y = -r[1] - L
        f = (a/4.)*(1./(np.cosh(x))**2)*(np.tanh(y) + 1.)
    elif(r[0] < 0 and r[1] > 0):
        x = -r[0] - L
        y = r[1] - L
        f = -(a/4.)*(1./(np.cosh(x))**2)*(np.tanh(y) + 1.)
    elif(r[0] <= 0 and r[1] <= 0):
        x = -r[0] - L
        y = -r[1] - L
        f = -(a/4.)*(1./(np.cosh(x))**2)*(np.tanh(y) + 1.)
    
    return f

def dwallsy(r,param):
    L = param[0]/2. - 0.1*(param[0]/2.)
    a = param[1]
    
    
    if(np.abs(r[0]) > L or np.abs(r[1]) > L):
        f = 0.
    elif(r[0] >= 0 and r[1] >= 0):
        x = r[0] - L/1.5
        y = r[1] - L/1.5
        f = (a/4.)*(np.tanh(x) + 1.)*(1./(np.cosh(y))**2)
    elif(r[0] > 0 and r[1] < 0):
        x = r[0] - L
        y = -r[1] - L
        f = -(a/4.)*(np.tanh(x) + 1.)*(1./(np.cosh(y))**2)
    elif(r[0] < 0 and r[1] > 0):
        x = -r[0] - L
        y = r[1] - L
        f = (a/4.)*(np.tanh(x) + 1.)*(1./(np.cosh(y))**2)
    elif(r[0] <= 0 and r[1] <= 0):
        x = -r[0] - L
        y = -r[1] - L
        f = -(a/4.)*(np.tanh(x) + 1.)*(1./(np.cosh(y))**2)
    
    return f


#############################
    
def rect(r,param):
    x0 = param[0]
    y0 = param[1]
    V0 = param[2]
    Lx = param[3]
    Ly = param[4]
    
    x = r[0] - x0
    y = r[1] - y0
    
    f = (V0/2.)*(np.tanh(-Lx*(np.abs(x)-(Lx/2.))) + 1)*(np.tanh(-Ly*(np.abs(y)-(Ly/2.))) + 1)
    '''
    if(np.abs(x) < (Lx/2.) and np.abs(y) < (Ly/2.)):
        f = V0
    elif(x >= 0 and np.abs(y) < (Ly/2.)):
        f = (V0/2.) * (np.tanh(-V0*(x - (Lx/2.))) + 1)
    elif(x < 0 and np.abs(y) < (Ly/2.)):
        f = (V0/2.) * (np.tanh(V0*(x + (Lx/2.))) + 1)
    elif(np.abs(x) < (Lx/2.) and y >= 0):
        f = (V0/2.) * (np.tanh(-V0*(y - (Ly/2.))) + 1)
    elif(np.abs(x) < (Lx/2.) and y < 0):
        f = (V0/2.) * (np.tanh(V0*(y + (Ly/2.))) + 1)
    else:
        f = 0.
    '''
    return f

        
def drectx(r,param):
    x0 = param[0]
    y0 = param[1]
    V0 = param[2]
    Lx = param[3]
    Ly = param[4]
    
    x = r[0] - x0
    y = r[1] - y0
    
    if(np.abs(x)<0.00001 or np.abs(x)>50):
        f = 0.
        return f
    f = -(V0/2.)*(x/np.abs(x))*(((1./np.cosh(Lx*(np.abs(x)-(Lx/2.))))**2))*(np.tanh(-Ly*(np.abs(y)-(Ly/2.))) + 1)
    '''
    if(np.abs(x) < (Lx/2.) and np.abs(y) < (Ly/2.)):
        f = V0
    elif(x >= 0 and np.abs(y) < (Ly/2.)):
        f = -(V0/2.)*V0* (1./np.cosh(-V0*(x - (Lx/2.))))**2
    elif(x < 0 and np.abs(y) < (Ly/2.)):
        f = (V0/2.)*V0* (1./np.cosh(V0*(x + (Lx/2.))))**2
    elif(np.abs(x) < (Lx/2.) and y >= 0):
        f = 0.
    elif(np.abs(x) < (Lx/2.) and y < 0):
        f = 0.
    else:
        f = 0.
    '''
    return f

def drecty(r,param):
    x0 = param[0]
    y0 = param[1]
    V0 = param[2]
    Lx = param[3]
    Ly = param[4]
    
    x = r[0] - x0
    y = r[1] - y0
    
    if(np.abs(y)<0.00001 or np.abs(y)>50):
        f = 0.
        return f
    f = -(V0/2.)*(y/np.abs(y))*((1./np.cosh(Ly*(np.abs(y)-(Ly/2.))))**2)*(np.tanh(-Lx*(np.abs(x)-(Lx/2.))) + 1)
    '''
    if(np.abs(x) < (Lx/2.) and np.abs(y) < (Ly/2.)):
        f = V0
    elif(x >= 0 and np.abs(y) < (Ly/2.)):
        f = 0.
    elif(x < 0 and np.abs(y) < (Ly/2.)):
        f = 0.
    elif(np.abs(x) < (Lx/2.) and y >= 0):
        f = -(V0/2.)*V0* (1./np.cosh(-V0*(y - (Ly/2.))))**2
    elif(np.abs(x) < (Lx/2.) and y < 0):
        f = (V0/2.)*V0* (1./np.cosh(V0*(y + (Ly/2.))))**2
    else:
        f = 0.
        '''
    return f
        
        
    




def recta(r,param):
    x0 = param[0]
    y0 = param[1]
    A = 2*param[2]
    Lx = param[3]
    Ly = param[4]
    
    Tx = 2*Lx
    Ty = 2*Ly 
    fx = 1./Tx
    fy = 1./Ty
    delta = 0.01
    
    x = r[0] - x0
    y = r[1] - y0
    
    if(np.abs(x) <= (Lx/2.) and np.abs(y) <= (Ly/2.)):
        f = (A/2.)*(4./(np.pi)**2)*np.arctan(np.sin(2*np.pi*fx*(x + 1/(4*fx))/delta))*np.arctan(np.sin(2*np.pi*fy*(y + 1/(4*fy))/delta)) + (A/2.)
    else:
        f = 0.
        
    return f

        
def drectax(r,param):
    x0 = param[0]
    y0 = param[1]
    A = 2*param[2]
    Lx = param[3]
    Ly = param[4]
    
    Tx = 2*Lx
    Ty = 2*Ly 
    fx = 1./Tx
    fy = 1./Ty
    delta = 0.01
    
    x = r[0] - x0
    y = r[1] - y0
    
    if(np.abs(x) <= (Lx) and np.abs(y) <= (Ly)):
        f = (A/2.)*(4./(np.pi)**2)*2*np.pi*fx*delta*((np.cos(2*np.pi*fx*(x + 1/(4*fx))))/((np.sin(2*np.pi*fx*(x + 1/(4*fx))))**2 + delta**2))*np.arctan(np.sin(2*np.pi*fy*(y + 1/(4*fy))/delta))
    else:
        f = 0.
        
    return f

def drectay(r,param):
    x0 = param[0]
    y0 = param[1]
    A = 2*param[2]
    Lx = param[3]
    Ly = param[4]
    
    Tx = 2*Lx
    Ty = 2*Ly 
    fx = 1./Tx
    fy = 1./Ty
    delta = 0.01
    
    x = r[0] - x0
    y = r[1] - y0
    
    if(np.abs(x) <= (Lx) and np.abs(y) <= (Ly)):
        f = (A/2.)*(4./(np.pi)**2)*2*np.pi*fy*delta*((np.cos(2*np.pi*fy*(y + 1/(4*fy))))/((np.sin(2*np.pi*fy*(y + 1/(4*fy))))**2 + delta**2))*np.arctan(np.sin(2*np.pi*fx*(x + 1/(4*fx))/delta))
    else:
        f = 0.
        
    return f
    
    
    
    
    
    
    