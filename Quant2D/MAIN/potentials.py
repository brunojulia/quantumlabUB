

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
    
    for i in range (24,76):
        pot[24,i] = V0
        pot[75,i] = V0
        pot[i,24] = V0
        pot[i,75] = V0
        '''
        pot[25,i] = V0
        pot[76,i] = V0
        pot[i,25] = V0
        pot[i,76] = V0
        '''
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
    
    for i in range (0,n):
        pot[i,65] = V0
        pot[i,66] = V0
    for i in range (45,56):
        pot[i,65] = 0.
        pot[i,66] = 0.
        
    return pot


def doubleslit(r,param):
    V0 = param[0]
    x0 = param[1]
    y0 = param[2]
        
    xx = r[0] - x0 #ROW
    yy = r[1] - y0 #COLUMN
    
    n = 100
    
    pot = np.zeros((n,n))
    
    for i in range (0,n):
        pot[i,65] = V0
        pot[i,66] = V0
    for i in range (40,46):
        pot[i,65] = 0.
        pot[i,66] = 0.
    for i in range (55,61):
        pot[i,65] = 0.
        pot[i,66] = 0.
    
    
    return pot
