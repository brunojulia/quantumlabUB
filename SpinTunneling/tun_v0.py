from kivy.app import App
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.screenmanager import FadeTransition, SlideTransition
#from kivy.uix.popup import Popup
from functools import partial
from kivy.properties import ObjectProperty,ReferenceListProperty,\
    NumericProperty,StringProperty,ListProperty,BooleanProperty

import numpy as np
import matplotlib.pyplot as plt

class tun_v0App(App):
    def build(self):
        self.title='Spin Tunneling'
        return MyScreenManager()

class MyScreenManager(ScreenManager):
    def  __init__(self, **kwargs):
        super(MyScreenManager, self).__init__(**kwargs)
        self.get_screen('Tunneling').stpseudo_init()
#        self.get_screen("Game").gpseudo_init()
    pass

class StartingScreen(Screen):
    def __init__(self,**kwargs):
        super(StartingScreen,self).__init__(**kwargs)

    def transition_ST(self):
        """Transicio del starting a l Spin Tunneling"""

        stscreen = self.manager.get_screen('Tunneling')
#        stscreen.st_schedule_fired()  # Aixo es per a fer la animacio
        self.manager.transition = FadeTransition()
        self.manager.current = 'Tunneling'

class TunnelingScreen(Screen):
    def __init__(self,**kwargs):
        super(TunnelingScreen,self).__init__(**kwargs)

    def stpseudo_init(self):
        '''Aqui he d'escriure tot el programa amb les fent .selg cada vegada que escric el nom d'una variable'''
        pass

if __name__ == '__main__':
    tun_v0App().run()