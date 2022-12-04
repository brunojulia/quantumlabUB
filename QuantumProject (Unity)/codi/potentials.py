#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: annamoresoserra
"""
from numba.experimental import jitclass
from numba import float64


spec = [
    ('x', float64),               
    ('y', float64),
    ('xmax', float64), 
    ('ymax', float64),  
    ('m', float64),
    ('w', float64),     
]


@jitclass(spec)
class Potentials:
    def __init__(self, x, y, xmax, ymax, m, w): #m i w aniran fora
        # object attributes
        self.x = x
        self.y = y
        self.xmax = xmax
        self.ymax = ymax
        self.m = m
        self.w = w
    
        
    def Vh(self):
        """Harmonic oscilator potential."""
        if self.x >= self.xmax or self.y >= self.ymax or self.x <= -self.xmax or self.y <= -self.ymax:
            return 1E28
        else:
            return (0.5 * self.m * (self.w) * (self.x**2 + self.y**2))
    
    
    def Vfree(self):
        """Potential of a free particle"""
        if self.x >= self.xmax or self.y>=self.ymax or self.x<=-self.xmax or self.y<=-self.ymax:
            return 10000000000. 
        else:
            return 0.
        
    
    
        
 
     