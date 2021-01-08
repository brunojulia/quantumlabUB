# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 12:13:39 2020

@author: llucv
"""
from kivy.app import App 
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen 
from kivy.uix.screenmanager import FadeTransition, SlideTransition   
from kivy.uix.popup import Popup
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.core.text import LabelBase
from kivy.core.window import Window

import numpy as np
import matplotlib.pyplot as plt



class qf_timelineApp(App):
    def build(self):
        self.title='quantum physics timeline'
        return MyScreenManager()

class MyScreenManager(ScreenManager):
    def  __init__(self, **kwargs):
        super(MyScreenManager, self).__init__(**kwargs)
        self.get_screen('Discoveries').dcpseudo_init()
        self.get_screen('Young_slit').yspseudo_init()
    pass


if __name__ == '__main__':
    qf_timelineApp().run()