

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
    
def woods_box(r,param):
    V0 = param[0]
    R = param[1]
    a = param [2]
    x0 = param[3]
    y0 = param[4]
        
    xx = r[0] - x0 #ROW
    yy = r[1] - y0 #COLUMN
    
   # up = 0. #-1.*V0/(1.+np.exp((yy-R)/a))*xx/xx
   # down = 0. #+1.*V0/(1.+np.exp((yy-R)/a))*xx/xx
   # left = 0. #-1.*V0/(1.+np.exp((-1*xx-R)/a))
    right = -1.*V0/(1.+np.exp((xx-R)/a))
    
    return right+(yy-yy)

#############################
    
def barrier_x(r,param):
    V0 = param[0]
    x0 = param [1]
        
    xx = r[0] - x0 #ROW
    yy = r[1]  #COLUMN
    
    return V0*xx**2*yy/yy