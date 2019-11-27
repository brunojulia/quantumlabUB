

import numpy as np

#############################
    
def osc(r,param):
    x0 = param[0]
    y0 = param[1]
    w = param[2]
    m=1.
    
    xx=r[0]-x0 #ROW
    yy=r[1]-y0 #COLUMN
    
    return 0.5*m*w**2*(xx**2+yy**2) #MATRIX

def osc_nosym(r,param):
    x0 = param[0]
    y0 = param[1]
    wx = param[2]
    wy = param[3]
    m=1.
    
    xx=r[0]-x0 #ROW
    yy=r[1]-y0 #COLUMN
    
    return 0.5*m*(wx**2*xx**2+wy**2*yy**2) #MATRIX

#############################
    
def gauss(r,param):
    x0 = param[0]
    y0 = param[1]
    V0 = param[2]
    sig = param[3]
    
    xx = r[0] - x0  #ROW
    yy = r[1] - y0  #COLUMN
    
    return V0*np.exp(-(xx**2)/(2*sig**2))*np.exp(-(yy**2)/(2*sig**2))

#############################
    
def box(r,param): #Barrier in a box
    V0 = param[0]
    x0 = param[1]
    y0 = param[2]
        
    xx = r[0] - x0 #ROW
    yy = r[1] - y0 #COLUMN
    
    n = 100
    
    pot = np.zeros((n,n))
    
    for i in range (28,72):
        pot[28,i] = V0
        pot[70,i] = V0
        pot[i,28] = V0
        pot[i,70] = V0
        
        pot[29,i] = V0
        pot[71,i] = V0
        pot[i,29] = V0
        pot[i,71] = V0
        
    return  pot

#############################
    
def barrier_x(r,param):
    V0 = param[0]
    x0 = param [1]
        
    xx = r[0] - x0 #ROW
    yy = r[1]  #COLUMN
    
    return V0*xx**2*yy/yy

#############################

def singleslit(r,param):
    V0 = param[0]
    x0 = param[1]
    y0 = param[2]
        
    xx = r[0] - x0 #ROW
    yy = r[1] - y0 #COLUMN
    
    n = 100
    
    pot = np.zeros((n,n), dtype = 'float')
    
    for i in range (0,39):
        pot[i,65] = V0
        pot[i,66] = V0  
    for i in range (60,n):
        pot[i,65] = V0
        pot[i,66] = V0
    
    return pot


def doubleslit(r,param):
    V0 = param[0]
    x0 = param[1]
    y0 = param[2]
        
    xx = r[0] - x0 #ROW
    yy = r[1] - y0 #COLUMN
    
    n = 100
    
    pot = np.zeros((n,n))
    
    for i in range (0,41):
        pot[i,70] = V0
    for i in range (45,56):
         pot[i,70] = V0
    for i in range (60,n):
        pot[i,70] = V0
    
    return pot
