#kivy imports
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.garden.knob import Knob
from kivy.graphics import Rectangle, Color

from kivy.properties import ObjectProperty, NumericProperty
import math

from entangledexp import entangledEXP  #importa les funcions que tenen a veure amb l'experiment.

class entangledscreen(BoxLayout):
	n_label=ObjectProperty()  #ara n és una propietat de entangledscreen i farem self. bla bla
	label_s1=ObjectProperty()
	label_s2=ObjectProperty()
	s_label=ObjectProperty()
	def __init__(self, *args, **kwargs):
		super(entangledscreen, self).__init__()
		self.experiment = entangledEXP()

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
	pass

class entangledApp(App):

	""""""
	def build(self):
		return entangledscreen()

if __name__ == "__main__":
    app = entangledApp()
    app.run()