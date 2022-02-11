#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 10:49:19 2021

@author: lauraguerrarivas
"""

from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.floatlayout import FloatLayout
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

import matplotlib.pyplot as plt

#Definició del que vull definir
x=[1,2,3,4,5]
y=[1,2,3,4,4]
plt.plot(x,y)


class design(FloatLayout):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)

		box = self.ids.box
		box.add_widget(FigureCanvasKivyAgg(plt.gcf()))

	def alpha_value(self):
		self.alpha = float(self.ids.alpha_text.text)
		print(self.alpha)
		print('')
		print(semar(self,alpha))

	def press_H(self,button):
		pass

	def semar(self,alpha):
		return (alpha+1)







class MainApp(MDApp):
	def build(self):
		self.theme_cls.theme_style = "Dark"
		self.theme_cls.primary_palette = "BlueGray"
		Builder.load_file('design.kv')
		return design()


if __name__ == '__main__':
    MainApp().run()
