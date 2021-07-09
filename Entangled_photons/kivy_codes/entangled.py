import matplotlib
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas
from matplotlib.figure import Figure
from numpy import arange, sin, pi
from kivy.uix.behaviors import ButtonBehavior
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt

#kivy imports
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.garden.knob import Knob
from kivy.graphics import Rectangle, Color
from kivy.lang import Builder
from  kivy.uix.popup import Popup
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas,\
                                                NavigationToolbar2Kivy

from kivy.properties import ObjectProperty, NumericProperty, StringProperty
import math

from entangledexp import entangledEXP  #importa les funcions que tenen a veure amb l'experiment.



class TablePopup(Popup):
	g_rectangle = ObjectProperty()

	def __init__(self,*args, **kwargs):
		super(TablePopup, self).__init__(*args, **kwargs)

class TrueScreen(ScreenManager):
	# def __init__(self, **kwargs):
	# 	super(TrueScreen, self).__init__()
	pass

class EntangledScreen(Screen):
	n_label=ObjectProperty()  #ara n és una propietat de entangledscreen i farem self. bla bla
	label_s1=ObjectProperty()
	label_s2=ObjectProperty()
	s_label=ObjectProperty()
	table_checkbox=ObjectProperty(CheckBox())
	table_popup=ObjectProperty()
	graph_checkbox=ObjectProperty(CheckBox())
	select_button=ObjectProperty()
	delete_button=ObjectProperty()
	plot_btn=ObjectProperty()
	clear_btn=ObjectProperty()
	b1_label=ObjectProperty()
	b2_label=ObjectProperty()
	b1_val=NumericProperty()
	b2_val=NumericProperty()
	kwinput=ObjectProperty()
	knoblay=ObjectProperty()

	def __init__(self,angle_count = 0,**kwargs):
		super(EntangledScreen, self).__init__()
		self.angle_count = angle_count
		self.kwinput = False
		self.experiment = entangledEXP()
		self.table_checkbox.bind(active=self.on_checkbox_Active)#lliga la checkbox amb la funció
		self.graph_checkbox.bind(active=self.on_graph_checkbox_Active)  # lliga la checkbox amb la funció

	def add_photons(self,a):
		self.experiment.addphotons(n=self.experiment.n+a) #suma 1000 als fotons a llençar
		self.n_label.text=str(self.experiment.n)

	def runexp(self):
		alpha = int(self.label_s1.text)*math.pi/180#convertim a radians i assignem els parametres per poder fer l'experiment
		beta = int(self.label_s2.text)*math.pi/180
		self.experiment.photons=int(self.n_label.text)

		#table1 = self.experiment.expqua(alpha, beta)
		s=self.experiment.scalc(alpha, beta)
		sigma=self.experiment.sigma(alpha, beta)
		print(s,"±",sigma)
		rounder=sigma
		factorcounter=0
		while rounder<1:
			rounder=rounder*10
			factorcounter+=1

		sr=round(s,factorcounter)
		sigmar=round(sigma, factorcounter)
		self.s_label.text='[font=Digital-7][color=000000][size=34] S='+str(sr)+'[/font]'+'±'+'[font=Digital-7]'+str(sigmar)+'[/color][/font][/size]'
		return(sr," ± ",sigmar)

	def activate_txtin_1(self):
		self.label_s1.disabled=True

	def open_table_popup(self):
		'''opens popup window'''
		popuplayout= TablePopup()# es un float layout q posem com a content al popup

		self.table_popup= TablePopup()
		self.table_popup.open()

	def close(self):
		self.table_popup.dismiss()
		self.table_checkbox.active = False  # reseteja la chkbox

	def on_checkbox_Active(self, checkboxInstance, isActive):
		if isActive:
			self.open_table_popup()

	def on_graph_checkbox_Active(self, checkboxInstance, isActive):
		self.kwinput = False
		if isActive:
			self.select_button.disabled = False
			self.delete_button.disabled = False
			self.clear_btn.disabled = False
			self.plot_btn.disabled = False
		if isActive==False:
			self.select_button.disabled = True
			self.delete_button.disabled = True
			self.clear_btn.disabled = True
			self.plot_btn.disabled = True
			self.b1_label.disabled = True
			self.b2_label.disabled = True
			self.angle_count = 0

	def select_angle(self):
		if self.angle_count < 2:
			self.angle_count+=1
		#  if the angles changed it replots the graph
		self.manager.get_screen('GS').changed_angles = True

		if self.angle_count==1:
			self.delete_button.disabled = False
			if not self.kwinput:
				self.b1_label.text = self.label_s2.text
				self.b1_val = float(self.label_s2.text)

			else:
				self.b1_val = float(self.b1_label.text)

		if self.angle_count == 2:
			self.delete_button.disabled = False
			#self.select_button.disabled = True
			if not self.kwinput:
				self.b2_label.text = self.label_s2.text
				self.b2_val = float(self.label_s2.text)
			else:
				self.b2_val = float(self.b2_label.text)
		print(self.angle_count)
		self.kwinput = False
	def delete_angle(self):

		if self.angle_count>0:
			if self.angle_count == 1:
				self.b1_label.text = ' '
				self.b1_val = 0
				self.delete_button.disabled=True
				self.select_button.disabled = False
			if self.angle_count == 2:
				print(self.angle_count)
				self.b2_label.text = ' '
				self.b2_val = 0
				self.select_button.disabled = False
			self.angle_count-=1
			print(self.angle_count)
		self.kwinput = False


	def clear_angles(self):
		self.b1_label.text = ' '
		self.b1_val = 0
		self.b2_label.text = ' '
		self.b2_val = 0
		self.delete_button.disabled = True
		self.select_button.disabled = False
		self.angle_count = 0

	pass

# AngleKnob is a knob with properties from the button such as on_release.

class AngleKnob(ButtonBehavior, Knob):

	pass
############################################ Graph Layout ################################################################

class GraphScreen(Screen):
	mainlay = ObjectProperty()
	canv=ObjectProperty()
	#if the angles are changed sets to true
	changed_angles=ObjectProperty()
	def __init__(self, *args, **kwargs):
		super(GraphScreen, self).__init__(*args, **kwargs)
		self.exitbtn = Button(size_hint = (1, 0.05),text = 'Go Back')
		self.exitbtn.bind(on_release=self.go_back)
		self.mainlay.add_widget(self.exitbtn, index=0)
		self.changed_angles = True
	def get_graph(self):
		fig = plt.figure()
		ax = Axes3D(fig)
		if self.changed_angles == True:
			self.manager.get_screen('ES').b1_val = float(self.manager.get_screen('ES').b1_label.text)
			self.manager.get_screen('ES').b2_val = float(self.manager.get_screen('ES').b2_label.text)
			self.manager.get_screen('ES').angle_count = 2
			self.manager.get_screen('ES').experiment.b1 = self.manager.get_screen('ES').b1_val * math.pi / 180
			self.manager.get_screen('ES').experiment.b2 = self.manager.get_screen('ES').b2_val * math.pi / 180

			(alphalist,betalist)=self.manager.get_screen('ES').experiment.sweepS()
			scalcvec = np.vectorize(self.manager.get_screen('ES').experiment.scalc)

			X, Y = np.meshgrid(alphalist, betalist, sparse=True)
			Z = scalcvec(X, Y)

			mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
			mappable.set_array(Z)

			ax.plot_surface(X, Y, Z, cmap=mappable.cmap, linewidth=0.01)

			ax.set_xlabel('Alpha (rad)')
			ax.set_ylabel('Beta (rad)')
			ax.set_zlabel('S')
			mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
			mappable.set_array(Z)
			cbar = fig.colorbar(mappable, shrink=0.5)
			cbar.set_label('S', rotation=0)

			self.canv = FigureCanvas(fig)
		self.add_plot()

	def add_plot(self):
		self.mainlay.add_widget(self.canv, index = 1)
		self.manager.current = 'GS'

	def go_back(self,instance):
		self.mainlay.remove_widget(self.canv)
		self.manager.current = 'ES'
		self.changed_angles = False
	pass

kv = Builder.load_file("entangled.kv")



class MainApp(App):

	#MS = EntangledScreen()

	def build(self):
		sm = TrueScreen()
		sm.add_widget(GraphScreen(name='GS'))
		sm.add_widget(EntangledScreen(name='ES'))
		sm.current = 'ES'
		return sm

if __name__ == "__main__":
    app = MainApp()
    app.run()