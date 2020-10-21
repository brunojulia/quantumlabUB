# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 19:24:43 2020

@author: josep
"""

import kivy
import numpy as np

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget
from kivy.uix.scrollview import ScrollView

global matrix
matrix =np.empty((2,2), str)
#print(matrix)


class CCell(FloatLayout):
    def add_element(self):
        button= Button(text="test2")
        self.add_widget(button)
    pass

class CRow(FloatLayout):
    v2 =ObjectProperty(None)
    def show(self):
        if self.v2.visible == False: 
            self.v2.visible = True
        elif self.v2.visible == True: 
            self.v2.visible = False
    pass

class CGrid(FloatLayout):
    grid = ObjectProperty(None)
    #row1= ObjectProperty(None)
    def add_element(self):
        for i in range(2):
            button= Button(text="test")
            self.grid.add_widget(button)
        return self.grid
    def add_element2(self, row,col):
        self.ids[row].ids[col].clear_widgets()
        button2= Button(text="test2",size_hint= (None, None), size= (70,70), pos_hint= {"x":0, "y":0})
        self.ids[row].ids[col].add_widget(button2)
    def add_element3(self):
        global matrix
        print(matrix)
        matrix=np.append(matrix, np.empty((2,1), str), axis=1)
        print(matrix)
        for i in range(2):
            test=Button(text="new",size_hint= (None, None), size= (70,70), pos= (self.ids[str(i)].x + 140, self.ids[str(i)].y))
            matrix[i][2]= CCell(id='cell',pos= (self.width+140, 0))
            self.ids[str(i)].add_widget(test)
        return matrix
            

    

class QComp(App):
    def build(self):
        return WindowManager()
        
class WindowManager(ScreenManager):
     pass


class Screen1(Screen):
    btn = ObjectProperty(None)
    cgrid = ObjectProperty(None)
    pass
        
    

class Screen2(Screen):
    pass


        

if __name__ =='__main__':
    QComp().run()