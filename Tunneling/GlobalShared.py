# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:53:34 2020

@author: usuari
"""

import numpy as np
import random

prob = 1.0
bonus = 1.0

#també escrivim la matriu d'amplades en aquest document
list_amp = np.zeros(10)
for i in range(0,10):
    list_amp[i] = 0.2 + i*0.1
    
#construim la matriu d'amplades de les barreres verticals
barr_vert = np.zeros((10,9,3))
barr_hor = np.zeros((9,10,3))

for i in range(0,10):
    for j in range(0,9):
        for k in range(0,3):
            barr_vert[i,j,k] = random.choice(list_amp)
        
for i in range(0,9):
    for j in range(0,10):
        for k in range(0,3):
            barr_hor[i,j,k] = random.choice(list_amp)
        
