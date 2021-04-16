#kivy imports
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox

from kivy.properties import ObjectProperty, NumericProperty
import math

from entangledexp import entangledEXP  #importa les funcions que tenen a veure amb l'experiment.

class entangledscreen(BoxLayout):
	n_label=ObjectProperty()  #ara n és una propietat de entangledscreen i farem self. bla bla
	label_s1=ObjectProperty()
	label_s2=ObjectProperty()
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
		print(s)
		return(s)
	pass

class entangledApp(App):

	""""""
	def build(self):
		return entangledscreen()

if __name__ == "__main__":
    app = entangledApp()
    app.run()