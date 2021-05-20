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
        self.particle=(False, self.x,self.y)
        self.lenkey=0
        #self.array=np.array([[1,None,None]])
        self.hack=0
        
        #per acabar a la pantalla Alice i Bob
        self.acabar=False
        
        #Per provar
        self.publica=False
        self.publicat=False
        self.publiA=None
        self.publiB=None
        self.arr0f=None
        self.arr1f=None
    
    #self.array=array
        
    