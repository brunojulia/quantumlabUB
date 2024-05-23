import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Color
from kivy.uix.popup import Popup
from class_percolacio_quadrat import ClassPercolacioQuadrat


##########################################################################################################################
##########################################################################################################################
class ClusterWidget(Widget):
    #Mètode constructor de la classe
    def __init__(self, simulation=None, **kwargs):
        super().__init__(**kwargs)
        self.simulation = simulation
        self.bind(size=self.update_canvas,pos=self.update_canvas)

##########################################################################################################################

    def update_canvas(self,*args):
        self.canvas.clear()
        if self.simulation:
            with self.canvas:
                vertex_actius = self.simulation.busca_clusters()
                matriu_pintada = self.simulation.pintar_clusters(vertex_actius)
                n = len(matriu_pintada)
                cell_size = self.width/n
                for i in range(n):
                    for j in range(n):
                        color = (0, 0, 0) if matriu_pintada[i, j] == 0 else (matriu_pintada[i, j] / len(vertex_actius), 0, 1)
                        Color(*color)
                        Rectangle(pos=(j * cell_size, self.height - (i + 1) * cell_size), size=(cell_size, cell_size))


##########################################################################################################################
##########################################################################################################################

#part visual. Comença preguntant quina dimensió n volem i quina probabilitat p volem.
#Genera la matriu i la mostra
class ClusterApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        self.simulation = None
        self.cluster_widget = ClusterWidget()

        layout.add_widget(self.cluster_widget)

        controls = BoxLayout(size_hint_y=None, height='50dp')

        self.n_input = TextInput(text='10', multiline=False)
        self.p_input = TextInput(text='0.5', multiline=False)

        generate_button = Button(text='Generate')
        generate_button.bind(on_press=self.generate_clusters)

        #Botó per veure si hi ha percolació o no
        check_button = Button(text='Check Percolation')
        check_button.bind(on_press=self.check_percolation)

        controls.add_widget(Label(text='Size:'))
        controls.add_widget(self.n_input)
        controls.add_widget(Label(text='Probability:'))
        controls.add_widget(self.p_input)
        controls.add_widget(generate_button)
        #per verifica si hi ha percolació o no
        controls.add_widget(check_button)  # Agregar botón de verificación

        layout.add_widget(controls)

        return layout

##########################################################################################################################

    def generate_clusters(self, instance):
        n = int(self.n_input.text)
        p = float(self.p_input.text)
        self.simulation = ClassPercolacioQuadrat(n, p)
        self.cluster_widget.simulation = self.simulation
        self.cluster_widget.update_canvas()

##########################################################################################################################

    def check_percolation(self, instance):
        if self.simulation:
            perc_n = self.simulation.percola()
            if perc_n != -1:
                message = 'Ha percolat :)'
            else:
                message = 'No ha percolat :('
            popup = Popup(title='Percolació', content=Label(text=message), size_hint=(None, None), size=(400, 200))
            popup.open()
        else:
            popup = Popup(title='Error', content=Label(text='Has de generar una matriu abans!'), size_hint=(None, None), size=(400, 200))
            popup.open()

##########################################################################################################################
##########################################################################################################################

if __name__ == '__main__':
    ClusterApp().run()
