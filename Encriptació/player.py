# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:36:18 2021

@author: Marina
"""

"Class- Player"
'''Aquí hi haurà totes les característiques del jugador. Això de moment ho estic fent només
pels jugadors de "Multiplayer"'''

class Player():
    def __init__(self,player,x,y):
        self.nplayer=player
        self.x=x
        self.y=y
        self.direccio='z'
        self.ready= False
        
    
    #self.array=array
        
    