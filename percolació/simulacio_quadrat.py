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
    # Mètode constructor de la classe
    def __init__(self, simulation=None, **kwargs):
        super().__init__(**kwargs)
        self.simulation = simulation
        #Amb aquesta funció identificarem el cluster que ha percolat i el pintarem de blanc
        self.percolated_cluster = None
        self.bind(size=self.update_canvas, pos=self.update_canvas)

    ##########################################################################################################################

    def update_canvas(self, *args):
        self.canvas.clear()
        if self.simulation:
            with self.canvas:
                vertex_actius = self.simulation.busca_clusters()
                matriu_pintada = self.simulation.pintar_clusters(vertex_actius)
                n = len(matriu_pintada)
                # Así te aseguras de que los cuadrados de color no se salgan de la pantalla
                cell_size = min(self.width/n, self.height/n)

                # Offset per centrar els quadrats.
                x_offset = (self.width - n*cell_size)/2
                y_offset = (self.height - n*cell_size)/2

                for i in range(n):
                    for j in range(n):
                        #comprovem que tenim un índex de cluster de percolació inicialitzat (això ja implica que hi ha percolació al sistema)
                        if self.percolated_cluster is not None and matriu_pintada[i, j] == self.percolated_cluster:
                            color = (1, 1, 1)  #pintem de color blanc el cluster que ha percolat. Això es veu amb la segona condició del if

                        #aquí simplement estem pintant la matriu de negre si no es supera el llindar p i de colors si es supera
                        #(vèrtex no actiu i actiu respectivament)
                        else:
                            color = (0, 0, 0) if matriu_pintada[i, j] == 0 else (matriu_pintada[i, j] / len(vertex_actius), 0,1)
                        Color(*color)
                        Rectangle(pos=(x_offset + j*cell_size, y_offset + self.height - (i + 1)*cell_size),
                                  size=(cell_size, cell_size))

##########################################################################################################################
##########################################################################################################################

# part visual. Comença preguntant quina dimensió n volem i quina probabilitat p volem.
# Genera la matriu i la mostra
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

        # Botó per veure si hi ha percolació o no
        check_button = Button(text='Check Percolation')
        check_button.bind(on_press=self.check_percolation)

        controls.add_widget(Label(text='Size:'))
        controls.add_widget(self.n_input)
        controls.add_widget(Label(text='Probability:'))
        controls.add_widget(self.p_input)
        controls.add_widget(generate_button)
        # per verifica si hi ha percolació o no
        controls.add_widget(check_button)

        layout.add_widget(controls)

        return layout

    ##########################################################################################################################

    def generate_clusters(self, instance):
        n = int(self.n_input.text)
        p = float(self.p_input.text)
        self.simulation = ClassPercolacioQuadrat(n, p)
        self.cluster_widget.simulation = self.simulation
        #inicialitzem l'index del cluster percolat a None (si no hem comprovat la percolació
        #no podem saber si ha percolat o no)
        self.cluster_widget.percolated_cluster = None
        self.cluster_widget.update_canvas()

    ##########################################################################################################################

    def check_percolation(self, instance):
        if self.simulation:
            perc_n = self.simulation.percola()

            if perc_n != -1:
                message = 'Ha percolat :)'
                #Actualitzem el canvas per pintar de blanc el clsuter percolat
                self.cluster_widget.percolated_cluster = perc_n
                self.cluster_widget.update_canvas()
                #Mostrem el missatge per pantalla
                popup = Popup(title='Percolació', content=Label(text=message), size_hint=(None, None), size=(400, 200))
                popup.open()

            else:
                message = 'No ha percolat :('
                popup = Popup(title='Percolació', content=Label(text=message), size_hint=(None, None), size=(400, 200))
                popup.open()
        else:
            popup = Popup(title='Error', content=Label(text='Has de generar una matriu abans!'), size_hint=(None, None),
                          size=(400, 200))
            popup.open()

        print(perc_n)


##########################################################################################################################
##########################################################################################################################

if __name__ == '__main__':
    ClusterApp().run()
