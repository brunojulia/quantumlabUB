

import numpy as np


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
#    np.seterr(all='raise')
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
    
    sign1 = np.where(xr>0,1,-1)
    sign2 = np.where(yr>0,1,-1)
    try:
        f1 = -V0*((sign1*np.cos(theta)*np.exp((Rx+px)/a))/(a*(np.exp(Rx/a)+np.exp(px/a))**2))*(1/(1 + np.exp((py-Ry)/a)))
        f2 = -V0*((sign2*np.sin(theta)*np.exp((Ry+py)/a))/(a*(np.exp(Ry/a)+np.exp(py/a))**2))*(1/(1 + np.exp((px-Rx)/a)))
    except RuntimeWarning:
        f1 = 0.
        f2 = 0.
    except FloatingPointError:
        f1 = 0.
        f2 = 0.
    f = f1 + f2
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
    
    sign1 = np.where(xr>0,1,-1)
    sign2 = np.where(yr>0,1,-1)
    try:
        f1 = V0*((sign1*np.sin(theta)*np.exp((Rx+px)/a))/(a*(np.exp(Rx/a)+np.exp(px/a))**2))*(1/(1 + np.exp((py-Ry)/a)))
        f2 = -V0*((sign2*np.cos(theta)*np.exp((Ry+py)/a))/(a*(np.exp(Ry/a)+np.exp(py/a))**2))*(1/(1 + np.exp((px-Rx)/a)))
    except RuntimeWarning:
        f1 = 0.
        f2 = 0.
    except FloatingPointError:
        f1 = 0.
        f2 = 0.
    f = f1 + f2
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

    
def acceptreject(n,a,b,M,fun,param):
    numbersx = np.array([])
    numbersy = np.array([])
    while (numbersx.size<n and numbersy.size<n):
        x = np.random.uniform(a,b)
        y = np.random.uniform(a,b)
        p = np.random.uniform(0,M)
        if(fun(x,y,param)>p):
            numbersx = np.append(numbersx,x)
            numbersy = np.append(numbersy,y)
            
    return numbersx,numbersy

def groundstateosc(x,y,param):    
    m = param[0]
    k = param[1]
    w = np.sqrt(k/m)
    
    
    a = m*w*0.5
    
    f = (1/np.sqrt(np.pi))*np.exp(-a*(x**2+y**2))
    return f

def groundstateoscp(px,py,param):
    m = param[0]
    k = param[1]
    w = np.sqrt(k/m)
    
    
    a = m*w*0.5
    
    f = (1/np.sqrt(a))*np.exp(-(1/(4*a))*((px)**2+(py)**2))
    return f

def freepart(x,y,param):
    x0 = param[0]
    y0 = param[1]
    px0 = param[2]
    py0 = param[3]
    sig = param[4]
    
    r = (x-x0)**2+(y-y0)**2
    
    f = (1/np.sqrt(2*np.pi*sig**2))*np.exp(1j*(px0*x + py0*y))*np.exp(-r/(4*sig**2))
    f = np.abs(f)**2
    return f
   
def freepartp(px,py,param):
    x0 = param[0]
    y0 = param[1]
    px0 = param[2]
    py0 = param[3]
    sig = param[4]
    
    sigp = 1/(sig*2)
    
    f = (1/(np.sqrt(np.pi)))*np.exp(-(((px-px0)**2 + (py-py0)**2)/sigp)/4.)
    f = np.abs(f)**2
    return f
    
    
    