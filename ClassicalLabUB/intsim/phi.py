
import numpy as np

class Phi():
    
    def __init__(self):
        self.functions = np.array([])
        self.derivatives = np.array([])
    def add_function(self,fun,dfun,param):
        self.functions = np.append(self.functions,(fun,param))
#        self.functions = np.append(self.functions,param)
        self.derivatives = np.append(self.derivatives,(dfun,param))
#        self.derivatives = np.append(self.derivatives,param)
        return
    def val(self,x,y):
        value = 0
        r = (x,y)
        for i in range(0,self.functions.shape[0],2):
            value = value + self.functions[0](r,self.functions[i+1])
        return value
    def dval(self,x,y):
        value = 0
        r = (x,y)
        for i in range(0,self.derivatives.shape[0],2):
            value = value + self.derivatives[0](r,self.derivatives[i+1])
        return value
    
    
def linear(r,param):
    f = param[0]*r[0]
    return f

def dlinear(r,param):
    f = param[0]
    return f

pot = Phi()
a = pot.functions
pot.add_function(linear,dlinear,[1,2])
a = pot.functions
b = pot.functions[0]([5,2],pot.functions[1])
pot.add_function(linear,dlinear,[5,2])
c = pot.functions[2]([5,2],pot.functions[3])
d = pot.val(5,1)