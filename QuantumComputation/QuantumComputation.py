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
from kivy.core.window import Window
from kivy.input.motionevent import MotionEvent
from kivy.graphics import *
from kivy.factory import Factory
from kivy.uix.scatter import Scatter
from functools import partial




global matrix
global row_name
global multigates
matrix =np.empty((0,0), str)
row_name=[]
gates_list=['A', 'B', 'C', 'D', 'E']
gates_2={'D': 2, 'E': 3}
pressed=''
multigates=[]


cellsize=50
space=5
    

class Gate_mouse(Button):
    def __init__ (self, **kwargs):
        global pressed
        super(Gate_mouse, self).__init__(**kwargs)
        self.text= pressed
        self.opacity=0.6
        self.size_hint=(None,None)
        self.size=(cellsize*0.9, cellsize*0.9)
    

class Gate_circuit(Button):
    
    def __init__ (self, **kwargs):
        super(Gate_circuit, self).__init__(**kwargs)
        self.always_release=True
        self.opacity=0.8
        self.size_hint=(0.9, 0.9)
        self.size=(cellsize/2., cellsize/2.)
        self.background_down = self.background_normal
        self.halign='center'
        self.bind(
            on_press=self.create_clock,
            on_release=self.delete_clock)
        
    def create_clock(self,  *args):
        global callback
        callback = partial(self.long_press)
        Clock.schedule_once(callback, 0.3)
        

    def delete_clock(self,  *args):
        global callback
        global pressed
        global cgrid
        
        Clock.unschedule(callback)
        pressed=''
        cgrid.remove_widget(cgrid.scatter)
        
        
    def long_press(self, *largs):
        global cgrid
        global matrix
        global pressed
        pressed=self.text
        #cgrid.rows[self.matrix[0]].cols[self.matrix[1]].remove_widget(self)
        self.parent.remove_widget(self)
        matrix[self.matrix[0],self.matrix[1]]=''
        
        posx=self.x-cgrid.scroll.parent.scroll_x*(cgrid.scroll.width-cgrid.scroll.parent.width)
        posy=self.y-cgrid.scroll.parent.scroll_y*(cgrid.scroll.height-cgrid.scroll.parent.height)+cgrid.scroll.parent.y
        cgrid.scatter =Gate_mouse(pos=(posx,posy))
        cgrid.add_widget(cgrid.scatter)
        
        
    def on_touch_move(self, touch):
        cgrid.scatter.center=touch.pos
        
    


class Gate_panel(Button):
    def __init__ (self, **kwargs):
        super(Gate_panel, self).__init__(**kwargs)
        self.always_release=True
        self.bind(on_press=self.press)
        self.bind(on_release=self.release)
        self.size_hint=(None,None)
        self.size= (cellsize,cellsize)
        self.background_down = self.background_normal

    def press(self, *args):
        global pressed
        global cgrid
        pressed=self.text
        cgrid.scatter =Gate_mouse(pos=(-100,-100))
        cgrid.add_widget(cgrid.scatter)

    def release(self, *args):
        global pressed
        pressed=''
        cgrid.remove_widget(cgrid.scatter)
        
    def on_touch_move(self, touch):
        cgrid.scatter.center=touch.pos
    
    

class GatesGrid(GridLayout):
    def __init__(self, **kwargs):
        super(GatesGrid, self).__init__(**kwargs)
        self.rows=1
        self.spacing=10,10
      
        for i in gates_list:
            self.add_widget(Gate_panel(text=i))
            
        

class CCell(FloatLayout):
    def __init__ (self, **kwargs):
        super(CCell, self).__init__(**kwargs)
        
        
    
    def on_touch_up(self, touch):
        global currentcell
        global matrix
        global row_name
        global pressed
        global multigate
        global rowsconnected
        
        scrollview=self.parent.parent.parent
        floatlay=self.parent.parent
        refx=self.x-scrollview.scroll_x*(floatlay.width-scrollview.width)
        refy=self.y-scrollview.scroll_y*(floatlay.height-scrollview.height)+scrollview.y
        
        
        if (refy < touch.y <refy+cellsize) and (refx < touch.x <refx+cellsize):   
            if pressed != '':
                row=matrix.shape[0]-self.parent.parent.children.index(self.parent)-1
                col=matrix.shape[1]-self.parent.children.index(self)-1  
                
                if pressed == 'multigate':
                    
                    self.secondclick(row, col)
                    
                            
                elif pressed in gates_2:
                    if currentcell != '':
                        self.remove_widget(self.gate)
                    gate=Gate_circuit(font_size='12sp',text=(pressed+'\n'+row_name[row]+'-> ?'), pos=(self.x+0.05*cellsize, self.y+0.05*cellsize))
                    currentcell= pressed
                    self.gate=gate
                    self.add_widget(self.gate)
                    self.gate.matrix=(row,col)
                    multigate=(row, col, pressed, gates_2[pressed])
                    rowsconnected=[row]
                    Clock.schedule_once(lambda dt:self.multigate(), 0)
                    
                
                else:
                    
                    if currentcell != '':
                        self.remove_widget(self.gate)
                    gate=Gate_circuit(text=pressed, pos=(self.x+0.05*cellsize, self.y+0.05*cellsize))
                    currentcell= pressed
                    self.gate=gate
                    
                    self.add_widget(self.gate)
                    self.gate.matrix=(row,col)

        
        super(CCell, self).on_touch_up(touch)
                    
    def multigate(self):
        global pressed
        pressed='multigate'
    
    def deletecurrent(self,col):
        global rowsconnected
        global matrix
        global cgrid
        global multigate
        global pressed
        
        for i in rowsconnected:
            cgrid.rows[i].cols[multigate[1]].remove_widget(cgrid.rows[i].cols[multigate[1]].gate)
            #print('before:',matrix[i][col])
            matrix[i][multigate[1]]=''
            #print(matrix[i][col])
        pressed=''
        
        
    def secondclick(self, row, col):
        global multigate
        global cgrid
        global matrix
        global currentcell
        global pressed
        global rowsconnected
        
        if (multigate[1]!=col) or (multigate[0]==row):
            Clock.schedule_once(lambda dt:self.deletecurrent(col), 0.05)
            
        else:
            if currentcell != '':
                self.remove_widget(self.gate)
            rowsconnected.append(row)
            #gatetext=multigate[2]+'\n'+row_name[multigate[0]]+'->'+row_name[row]
            gatetext=multigate[2]+'\n'+str(rowsconnected)
            gate=Gate_circuit(font_size='12sp',text=(gatetext), pos=(self.x+0.05*cellsize, self.y+0.05*cellsize))
            currentcell= multigate[2]
            self.gate=gate
            self.add_widget(self.gate)
            self.gate.matrix=(row,col)
            
            pressed=''
            
            multigate=(multigate[0], multigate[1], multigate[2],multigate[3]-1)
            print(multigate[3])
            
            if multigate[3] > 1:
                pressed='multigate'
                #print(pressed)
                
            print('rows',rowsconnected)
                
            
            
            
                    
       
                

class CRow(FloatLayout):
    global matrix
    def __init__(self, **kwargs):
        global row_num
        super(CRow, self).__init__(**kwargs)
        self.cols=[]
        row_num=matrix.shape[0]
        self.reini(-1)
        
    def refresh_row(self, num):
        global currentcell
        self.clear_widgets()
        self.reini(num)
        for i in range(len(self.cols)):
            self.add_widget(self.cols[i])
            if matrix[num][i] != '':
                    self.cols[i].remove_widget(self.cols[i].gate)
                    currentcell=matrix[num][i]
                    self.cols[i].gate=Gate_circuit(text=currentcell, pos=(self.cols[i].x+0.05*cellsize, self.y+0.05*cellsize))

                    self.cols[i].add_widget(self.cols[i].gate)
                    self.cols[i].gate.matrix=(num, i)
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
            if row_name[i+num] == ('q'+str(i+num+1)):
                row_name[i+num] = ('q'+str(i+num))
        self.parent.parent.parent.rows.pop(num)
        self.parent.parent.parent.refresh()
        self.parent.remove_widget(self)
        
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
        global cgrid
        super(CGrid, self).__init__(**kwargs)
        self.rows=[]
        for i in range(2):
            Clock.schedule_once(lambda dt:self.add_row(), 0.5)
        for i in range(12):
            Clock.schedule_once(lambda dt:self.add_col(), 0.5)
        Clock.schedule_once(lambda dt:self.resize_scroll(), 1.2)
            
        Window.bind(on_resize=Clock.schedule_once(lambda dt:self.resize_scroll(), 0))
        Window.bind(on_resize=Clock.schedule_once(lambda dt:self.set_scroll(), 0))
        cgrid=self
        
    def on_touch_up(self, touch):
        global matrix
        global currentcell
        if self.collide_point(*touch.pos):
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    currentcell=matrix[i][j]
                    self.rows[i].cols[j].on_touch_up(touch)
                    matrix[i][j]=currentcell
            #super(CGrid, self).on_touch_up(touch)
        
    def add_col(self): #Adds column to matrix and rows
        global matrix
        matrix=np.append(matrix, np.empty((matrix.shape[0],1), str), axis=1)
        for i in range(matrix.shape[0]):
            self.rows[i].add_cell()
        self.resize_scroll()
        return matrix

    def refresh(self): #Refresh buttons
        self.resize_scroll()
        for i in range(matrix.shape[0]):
            self.rows[i].pos= (0, self.scroll.y+(cellsize+space)*(matrix.shape[0]-i-1))
            self.rows[i].refresh_row(i)
    
    def remove_col(self): #Remove column from matrix and rows
        global matrix
        if matrix.shape[1] != 0:
            matrix=np.delete(matrix, -1, axis=1)
        for i in range(matrix.shape[0]):
            self.rows[i].remove_cell()
        self.resize_scroll()
        return matrix
    
    def add_row(self): #Add row
        global matrix
        global row_name
        matrix=np.append(matrix, np.empty((1,matrix.shape[1]), str), axis=0)
        row_name.append('q'+str(len(row_name)))
        self.rows.append(CRow(pos= (0, self.scroll.y-(cellsize+space)*(matrix.shape[0]-1))))
        self.scroll.add_widget(self.rows[-1])
        for i in range(matrix.shape[1]):
            self.rows[-1].add_cell()
        self.refresh()
    
    def resize_scroll(self, *args):
        self.scroll.size=(cellsize*(matrix.shape[1]+1.5)+70,(cellsize+space)*(matrix.shape[0]+0.5))
    
    def set_scroll(self): #Necessary to avoid problems when resizing
        self.scroll.parent.scroll_x=0.0
        self.scroll.parent.scroll_y=1.0
    
    def resize_panels(self, prob, gates):
        width=0.7 if prob else 1
        height=0.62 if gates else 0.92
        self.parent.parent.size_hint=(width, height)
        Clock.schedule_once(lambda dt:self.resize_scroll(), 0)
        Clock.schedule_once(lambda dt:self.set_scroll(), 0)

    def add_element2(self, *args):
        global pressed
        global matrix
        print(pressed)
        print(matrix)
     

class QComp(App):
    def build(self):
        return WindowManager()
        
class WindowManager(ScreenManager):
     pass


class Screen1(Screen):
    #cgrid = ObjectProperty(None)
    pass


        
    

class Screen2(Screen):
    pass


        

if __name__ =='__main__':
    QComp().run()