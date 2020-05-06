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

############
class PaquetApp(App):
    """Inicialitzem la aplicació"""
    
    title='PaquetApp'
    def build(self):
        return MyScreenManager()

class MyScreenManager(ScreenManager):
    "Activarà la pantalla d'inici.Manetja les screens, claro..."
    
    def __init__(self,**kwargs):
        super(MyScreenManager,self).__init__(**kwargs)
        self.get_screen('paquet').gpseudo_init()
        self.get_screen('paquet').g_schedule_fired()

class PaquetScreen(Screen):
    "Això és la part més important de la app. Introduïrem la pantalla inicial"
    def __init__(self,**kwargs):
        "Unim la clase amb la paraula self"
        super(PaquetScreen,self).__init__(**kwargs)
        
    def gpseudo_init(self):
        "Iniciem el primer dibuix de l'aplicació"
        #Parametres utiltizats
        self.L=3
        self.tb=0.1
        self.ta=0
        self.dx=0.03
        self.dt=0.01
        self.Nx=np.int((2*self.L)/self.dx)
        self.Nt=np.int((self.tb-self.ta)/self.dt)
        self.m=1
        self.hbar=1
        #Moments inicials que no es corresponen amb els finals
        self.px=0
        self.py=0
        self.i=0
        #El que utiltizarem per dibuixar
        
                       
        
        self.normavec=np.load('norm.npy')
    
        
        #Figura
        self.main_fig=plt.figure()
        #L'associem amb kivy
        self.main_canvas=FigureCanvasKivyAgg(self.main_fig)
        self.box1.add_widget(self.main_canvas)
        #Plot principal
        self.visu=plt.subplot()
        self.visu_im=self.visu.imshow(self.normavec[:,:,0],origin={'lower'},
                                      extent=(-self.L,self.L,-self.L,self.L))
        #Dibuixem tot lo dit
        self.main_fig.tight_layout()
        self.main_canvas.draw()
        
        #Afegim uns butons. Cada un augmenta una cosa
        self.box2.add_widget(Button(text='px+1',on_press=partial(self.changep,1,0)))
        self.box2.add_widget(Button(text='px-1',on_press=partial(self.changep,-1,0)))
        self.box2.add_widget(Button(text='py+1',on_press=partial(self.changep,0,1)))
        self.box2.add_widget(Button(text='py-1',on_press=partial(self.changep,0,-1)))
        self.box2.add_widget(Button(text='Compute',on_press=partial(self.compute)))
        self.label=Label(text='px={}'.format(self.px)+'\n'+
                                   'py={}'.format(self.py))
        self.box2.add_widget(self.label)
        
        self.pause_state=True
        
    def g_schedule_fired(self):
        "Ara, definim l'update, a partir d'aqui farem l'animació"
        self.schedule=Clock.schedule_interval(self.plotpsiev,1/60.)
    
    def plotpsiev(self,dt):
        "Update function that updates psi plot"
        
        if self.pause_state==False:
            self.i+=1
            if(self.i<self.Nt):
        
                
                #Animation
                #Lanimació consisteix en elimanr primer el que teniem abans
                self.visu_im.remove()
                #Posar el nou
                self.visu_im=self.visu.imshow(self.normavec[:,:,self.i],origin={'lower'},
                                              extent=(-self.L,self.L,-self.L,self.L)) 
                #I utilitzar-lo
                self.main_canvas.draw()
            if(self.i>=self.Nt):
                self.i=0
                self.pause_state=True
            
             
    
    #Afeim això que ens canvia px e py
    def changep(self,add_px,add_py,*largs):
        self.px+=add_px
        self.py+=add_py
        print(self.px,self.py)
        self.box2.remove_widget(self.label)
        self.label=Label(text='px={}'.format(self.px)+'\n'+
                                   'py={}'.format(self.py))
        self.box2.add_widget(self.label)
        
    #Aquesta funció s'encarreguera de calcular i activar tot un seguïnt 
    #tot un següit de paramètres
    
    def compute(self,*largs):
        psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py) 
                             for i in range(self.Nx+1)]
                       for j in range(self.Nx+1)],dtype=np.complex128)
        psivec,normas,tvec,Vvec=ck.Crannk2D(-self.L,self.L,-self.L,self.L,
                                                self.ta,self.tb,self.Nx,self.Nx,
                                                self.Nt,ck.Vharm,self.hbar,self.m,
                                                psi0)
        self.normavec=normas
 
        self.pause_state=False
                                             
if __name__=='__main__':
    PaquetApp().run()
    
    