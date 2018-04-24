"""
Jan Albert Iglesias
16/4/2018
"""

import numpy as np
import random as rnd
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

cldt = 1.0/60
qudt = 2.0

class Variable(Widget):
    pass
""" Faig servir aquesta variable de manera tant barroera pk no em
deixa fer servir el classical.width correctament. I no puc posar la
bola al mig. Així que el que faig és fer-ho amb una variable que
comparteixo amb el kivy. Allà si que el classical_id.width l'entén bé.
"""

class Ball(Widget):
    pass

class gaussian(Image):
    pass

class wavefunScreen(Screen):
    pass

class kivycat(BoxLayout):
    def __init__(self, **kwargs):
        super(kivycat, self).__init__(**kwargs)
        #No sé per què però no em deixa definir la posició des d'aquí. Pos es refereix a baix a l'esquerra.
        #Provar de fer una variable i donar valors a aquesta variable. Mantenint allà la mida relativa.
        #N'hi ha prou definint el id: a la boxlayout i no al widget. No va tan bé.
        self.variable1.size = 0,0
        self.variable2.size = 0,0
        self.switch = 1
        self.t = 0

    def classicupdate(self,dt):
        self.t = self.t + dt
        self.variable1.size = 70*int(100*np.sin(self.t))/100, 0


            #Provar de fer la mida relativa a la caixa. Amb la pilota no, que me l'aixafarà.

    def quantumupdate(self,dt):
        self.switch = self.switch*(-1)
        if self.switch > 0:
            self.gaussian.color = (1,1,1,0) #To hide the gaussian image, I change the opacity.
            self.eye.color = (1,1,1,1)
            self.variable2.width = 100*rnd.random()
        else:
            self.gaussian.color = (1,1,1,1)
            self.eye.color = (1,1,1,0) #It also hides the marker because it inherits from the above conditions.


sm = ScreenManager()
sm.add_widget(wavefunScreen(name='wfScreen'))
sm.add_widget(wavefunScreen(name='wfScreen'))

class kivycatApp(App):
    def build(self):
        self.title = "Schrödinger's cat"
        kat = kivycat()
        Clock.schedule_interval(kat.classicupdate, cldt)
        Clock.schedule_interval(kat.quantumupdate, qudt)
        return kat

if __name__ == "__main__":
    kivycatApp().run()
