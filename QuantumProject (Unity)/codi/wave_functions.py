#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:12:43 2022

@author: annamoresoserra
"""
import math
import numpy as np
from numba.experimental import jitclass
from numba import float64


spec = [
    ('x', float64),               
    ('y', float64),
    ('x0', float64), 
    ('y0', float64),       
]


@jitclass(spec)
class WaveFunction:
    def __init__(self, x, y, x0, y0):
        # object attributes
        self.x = x
        self.y = y
        self.x0 = x0
        self.y0 = y0
        
    
        
    def psi_harm0(self):
        c = ((1) / (math.pi))**(1./2.)
        return c * (np.exp(- (((self.x-self.x0)**2 + (self.y-self.y0)**2)) / (2.)))
    
   
    
    def psi_harm1(self):
        c = 2./((math.pi)**(1./2.))
        return c * (np.exp(-(((self.x-self.x0)**2 + (self.y-self.y0)**2)) / (2.0))) * (self.x-self.x0) * (self.y-self.y0) 


     
    def psi_harm2(self):
        c = 1. / (((math.pi)**(1./2.)) * 8.)
        return c * (np.exp(-(((self.x-self.x0)**2 + (self.y-self.y0)**2)) / (2.0))) * (4.0 * ((self.x - self.x0) ** 2) - 2.0) * (4.0 * ((self.y-self.y0) ** 2) - 2.0)


    
    def psigauss(self, sig, px, py): # pensar com afegir l'opcio de canviar sigma
        return (1.0 / (math.sqrt(2.0 * math.pi * sig))) * np.exp(1j * (px * (self.x - self.x0) + py * (self.y - self.y0)) - (1.0 / (4.0 * sig)) * ((self.x - self.x0) ** 2 + (self.y - self.y0) ** 2))
