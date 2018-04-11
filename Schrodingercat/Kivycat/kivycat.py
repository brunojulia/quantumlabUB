"""
Jan Albert Iglesias
10/4/2018
"""

import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.clock import Clock

class Variable(Widget):
    pass
""" Faig servir aquesta variable de manera tant barroera pk no em
deixa fer servir el classical.width correctament. I no puc posar la
bola al mig. Així que el que faig és fer-ho amb una variable que
comparteixo amb el kivy. Allà si que el classical_id.width l'entén bé.
"""

class Ball(Widget):
    pass

class kivycat(BoxLayout):
    def __init__(self, **kwargs):
        super(kivycat, self).__init__(**kwargs)
        #No sé per què però no em deixa definir la posició des d'aquí. Pos es refereix a baix a l'esquerra.
        #Provar de fer una variable i donar valors a aquesta variable. Mantenint allà la mida relativa.
        #N'hi ha prou definint el id: a la boxlayout i no al widget. No va tan bé.
        self.variable.size = 0,0
        self.t = 0

    def update(self,dt):
        self.t = self.t + dt
        self.variable.size = 100*int(100*np.sin(self.t))/100,0

class kivycatApp(App):
    def build(self):
        self.title = "Schrödinger's cat"
        kat = kivycat()
        Clock.schedule_interval(kat.update, 1.0/60)
        return kat

if __name__ == "__main__":
    kivycatApp().run()
