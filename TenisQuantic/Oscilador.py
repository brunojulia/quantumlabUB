# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adrià Bravo Vidal)s
"""

import cranknicolsonbe as ck
from kivy.app import App 
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen 
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.widget import Widget
import numpy as np
import matplotlib.pyplot as plt 
from functools import partial
from kivy.properties import ObjectProperty,ReferenceListProperty,NumericProperty,StringProperty
from numpy.polynomial import hermite as hermite
import math
############


class OsciladorApp(App):
    title='Oscilador'
    def build(self):
        
        return MyScreenManager()
    

class MyScreenManager(ScreenManager):
    "Activarà la pantalla d'inici.Manetja les screens, claro..."
    
    def __init__(self,**kwargs):
        super(MyScreenManager,self).__init__(**kwargs)
        self.get_screen('oscilador').gpseudo_init()

        
class OsciladorScreen(Screen):
    "Això és la part més important de la app. Introduïrem la pantalla inicial"
    def __init__(self,**kwargs):
        "Unim la clase amb la paraula self"
        super(OsciladorScreen,self).__init__(**kwargs)
        
    def gpseudo_init(self):
        "Iniciem el primer dibuix de l'aplicació"
        #Parametres utiltizats
        self.L=3
        self.dx=0.03
        self.Nx=np.int((2*self.L)/self.dx)
        self.m=1
        self.hbar=1
        #Moments inicials que no es corresponen amb els finals
        self.nx=0
        self.ny=0
        self.i=0
        #Preparem el paquet inicial
        psi0vec=np.array([[psi0harm(-self.L+np.float(i)*self.dx,-self.L+np.float(j)*self.dx,1.,
                                    self.m,self.hbar,self.nx,self.ny)  
                    for i in range(self.Nx+1)] for j in range(self.Nx+1)])
        self.normavec=norma(psi0vec,self.Nx)
        #Figura
        self.main_fig=plt.figure()
        #L'associem amb kivy
        self.main_canvas=FigureCanvasKivyAgg(self.main_fig)
        self.box1.add_widget(self.main_canvas)
        #Plot principal
        self.visu=plt.subplot()
        self.visu.set_title('Estats fonamentals oscilador harmonic(w=1)')
        self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                      extent=(-self.L,self.L,-self.L,self.L))
        #Dibuixem tot lo dit
        self.main_fig.tight_layout()
        self.main_canvas.draw()
        
         #Afegim uns butons. Cada un augmenta una cosa
        self.box2.add_widget(Button(text='nx+1',on_press=partial(self.changen,1,0)))
        self.box2.add_widget(Button(text='nx-1',on_press=partial(self.changen,-1,0)))
        self.box2.add_widget(Button(text='ny+1',on_press=partial(self.changen,0,1)))
        self.box2.add_widget(Button(text='ny-1',on_press=partial(self.changen,0,-1)))
        self.label=Label(text='nx={}'.format(self.nx)+'\n'+
                                   'ny={}'.format(self.ny))
        self.box2.add_widget(self.label)
        
        #Afeim això que ens canvia nx e ny
    def changen(self,add_nx,add_ny,*largs):
        self.nx+=add_nx
        self.ny+=add_ny
        print(self.nx,self.ny)
        if (self.nx<0) or (self.ny<0):
            self.nx=0
            self.ny=0
        #Canviem el marcador
        self.box2.remove_widget(self.label)
        self.label=Label(text='nx={}'.format(self.nx)+'\n'+
                                   'ny={}'.format(self.ny))
        self.box2.add_widget(self.label)
        #Canviem la grafica, creem un nou paquet i el substituim
        self.visu_im.remove()
        psi0vec=np.array([[psi0harm(-self.L+np.float(i)*self.dx,-self.L+np.float(j)*self.dx,1.,
                                    self.m,self.hbar,self.nx,self.ny)  
                    for i in range(self.Nx+1)] for j in range(self.Nx+1)])
        self.normavec=norma(psi0vec,self.Nx)
        self.visu_im.show=self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                      extent=(-self.L,self.L,-self.L,self.L))
        
        self.main_canvas.draw()
    
def psi0harm(x,y,w,m,hbar,nx,ny):
    nx=np.int(nx)
    ny=np.int(ny)
    coefx=np.zeros(nx+1)
    coefx[nx]=1
    coefy=np.zeros(ny+1)
    coefy[ny]=1
    b=np.sqrt(hbar/(m*w))
    hermx=hermite.hermval(x/b,coefx)
    hermy=hermite.hermval(y/b,coefy)
    
    psi0=np.sqrt(1./((b**2)*(2**(nx+ny))*np.math.factorial(nx)*np.math.
                     factorial(ny)*np.pi))*np.exp(-0.5*((m*w)/hbar)*((x)**2+y**2))*(hermx)*(hermy)
    return psi0


#%
def norma(psi,Nx):
    return np.array([[np.real((psi[i,j])*np.conj(psi[i,j])) for j in range(Nx+1)] 
             for i in range(Nx+1)])

if __name__=='__main__':
    OsciladorApp().run()
    