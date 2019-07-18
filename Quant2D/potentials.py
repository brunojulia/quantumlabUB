

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
    
def barrierx(r,param):
    x0 = param[0]
    
    n=len(r[0])
    dx=abs(r[0][0,0]-r[0][0,1])
    
    f=np.zeros((n,n))
    for i in range(0,n+1):
        if (abs(i*dx-x0) < dx):
            f[i,:]=10**8
    
    return f
        
#############################
'''    
def dslit(r,param):
    x0 = param[0]
    y0 = param[1]
    d = param[2] #distance to obertures from y0
    c = param[3] #lenght of obertures
    
    n=len(r[0])
    dx=abs(r[0][0,0]-r[0][0,1])
    
    f=np.zeros((n,n))
    for i in range(0,n+1):
        if (abs(i*dx-x0) < dx):
            f[i,:]=10**5

    
    for j in range(0,n+1):
    
'''

    