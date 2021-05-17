#kivy imports
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.garden.knob import Knob
from kivy.graphics import Rectangle, Color
from  kivy.uix.popup import Popup

from kivy.properties import ObjectProperty, NumericProperty
import math

from entangledexp import entangledEXP  #importa les funcions que tenen a veure amb l'experiment.

class TablePopup(FloatLayout):
	g_rectangle = ObjectProperty()

	def __init__(self, *args, **kwargs):
		super(TablePopup, self).__init__(*args, **kwargs)

class entangledscreen(BoxLayout):
	n_label=ObjectProperty()  #ara n és una propietat de entangledscreen i farem self. bla bla
	label_s1=ObjectProperty()
	label_s2=ObjectProperty()
	s_label=ObjectProperty()
	table_checkbox=ObjectProperty()
	table_popup=ObjectProperty()

	def __init__(self, *args, **kwargs):
		super(entangledscreen, self).__init__()
		self.experiment = entangledEXP()
		self.table_checkbox.bind(active=self.on_checkbox_Active)#lliga la checkbox amb la funció

	def add_photons(self,a):
		self.experiment.addphotons(n=self.experiment.n+a) #suma 1000 als fotons a llençar
		self.n_label.text=str(self.experiment.n)

	def runexp(self):
		self.experiment.alpha=int(self.label_s1.text)*math.pi/180#convertim a radians i assignem els parametres per poder fer l'experiment
		self.experiment.beta = int(self.label_s2.text)*math.pi/180
		self.experiment.photons=int(self.n_label.text)

		table1 = self.experiment.expqua()
		s=self.experiment.scalc(table1)
		sigma=self.experiment.sigma(table1)
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

		self.table_popup= Popup(title='Table 1',content=popuplayout,size_hint=(1,1))
		self.table_popup.open()

	def close(self):
		self.table_popup.dismiss()
		self.table_checkbox.active = False  # reseteja la chkbox

	def on_checkbox_Active(self, checkboxInstance, isActive):
		if isActive:
			self.open_table_popup()
	pass
class AngleKnob(Knob):

	def __init__(self, **kwargs):
		self.screen = App.get_running_app().screen
	def on_touch_up(self, touch):
		if self.collide_point(touch.x,touch.y):
			self.screen.runexp()
		return super(AngleKnob,self).on_touch_up(touch)
	pass

class entangledApp(App):

	#hi ha un proboema quan poso això :(
	#screen = entangledscreen()
	def build(self):

		return entangledscreen()

if __name__ == "__main__":
    app = entangledApp()
    app.run()