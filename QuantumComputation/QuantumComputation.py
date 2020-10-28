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
from kivy.clock import Clock

global matrix
global row_name
matrix =np.empty((0,0), str)
row_name=[]


cellsize=50
space=5



class CCell(FloatLayout):
    def add_element(self):
        button= Button(text="test2")
        self.add_widget(button)
    pass



class CRow(FloatLayout):
    global matrix
    def __init__(self, **kwargs):
        global row_num
        super(CRow, self).__init__(**kwargs)
        self.cols=[]
        row_num=matrix.shape[0]
        self.reini(-1)
        
    def refresh_row(self, num):
        self.clear_widgets()
        self.reini(num)
        for i in range(len(self.cols)):
            self.add_widget(self.cols[i])
        return matrix
    
    def add_cell(self):
        self.cols.append(CCell(size_hint= (None, None), size= (cellsize,cellsize), pos_hint= {"y":0}, pos=(self.x+cellsize*(len(self.cols)+0.5)+70,0)))
        self.add_widget(self.cols[-1])
        
    def remove_cell(self):
        if len(self.cols) != 0:
            self.remove_widget(self.cols[-1])
            self.cols.pop(-1)
            
    def delete(self):
        global row_name
        global matrix
        num=row_name.index(self.name_label.text)
        matrix=np.delete(matrix, num, axis=0)
        row_name.pop(num)
        for i in range(len(row_name)-num):
            print(i, row_name[i+num],'q'+str(i+num+1))
            if row_name[i+num] == ('q'+str(i+num+1)):
                row_name[i+num] = ('q'+str(i+num))
        self.parent.parent.parent.rows.pop(num)
        self.parent.parent.parent.refresh()
        print(matrix.shape, row_name)
        self.parent.remove_widget(self)
        print(matrix.shape, row_name)
        
    def reini(self, num):
        global row_name
        qname=row_name[num]
        self.name_label=Label(text=qname,
                              color=(0,0,0,1),
                              halign='left',
                              texture_size=(30,30),
                              text_size=(40, None), 
                              pos_hint={"center_y":0.5}, 
                              pos= (25,0))
        self.add_widget(self.name_label)
        self.add_widget(Button(text="X", 
                               size_hint= (None, None), 
                               size= (self.height/2. , self.height/2.),
                               pos_hint= {"center_y": 0.5},
                               pos= (10,0),
                               on_release= lambda a:self.delete()))
        

class CGrid(FloatLayout):
    grid = ObjectProperty(None)
    scroll=ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super(CGrid, self).__init__(**kwargs)
        self.rows=[]
        for i in range(2):
            Clock.schedule_once(lambda dt:self.add_row(), 1)
        for i in range(12):
            Clock.schedule_once(lambda dt:self.add_col(), 1)
        
    def add_element(self):
        for i in range(2):
            button= Button(text="test")
            self.grid.add_widget(button)
        return self.grid
    
    def add_element2(self, row,col): #Modify cell (not from matrix)
        self.rows[row].cols[1].clear_widgets()
        button2= Button(text="test2",size_hint= (None, None), size= (cellsize,cellsize), pos_hint= {"x":0, "y":0})
        self.rows[row].cols[1].add_widget(button2)
        
    def add_col(self): #Adds column to matrix and rows
        global matrix
        matrix=np.append(matrix, np.empty((matrix.shape[0],1), str), axis=1)
        for i in range(matrix.shape[0]):
            self.rows[i].add_cell()
        self.resize()
        return matrix

    def refresh(self): #Refresh buttons
        self.resize()
        for i in range(matrix.shape[0]):
            self.rows[i].pos= (0, self.scroll.y+(cellsize+space)*i)
            self.rows[i].refresh_row(i)
        return matrix
    
    def remove_col(self): #Remove column from matrix and rows
        global matrix
        if matrix.shape[1] != 0:
            matrix=np.delete(matrix, -1, axis=1)
        for i in range(matrix.shape[0]):
            self.rows[i].remove_cell()
        self.resize()
        return matrix
    
    def add_row(self): #Add row
        global matrix
        global row_name
        matrix=np.append(matrix, np.empty((1,matrix.shape[1]), str), axis=0)
        row_name.append('q'+str(len(row_name)))
        self.rows.append(CRow(pos= (0, self.scroll.y+(cellsize+space)*(matrix.shape[0]-1))))
        self.scroll.add_widget(self.rows[-1])
        self.resize()
        for i in range(matrix.shape[1]):
            self.rows[-1].add_cell()
        return matrix 
    
    def resize(self):
        self.scroll.size=(cellsize*(matrix.shape[1]+1.5)+70,(cellsize+space)*(matrix.shape[0]+0.5))
    

class QComp(App):
    def build(self):
        return WindowManager()
        
class WindowManager(ScreenManager):
     pass


class Screen1(Screen):
    cgrid = ObjectProperty(None)
    pass
        

        
    

class Screen2(Screen):
    pass


        

if __name__ =='__main__':
    QComp().run()