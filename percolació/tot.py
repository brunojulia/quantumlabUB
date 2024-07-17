import random
import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Rectangle, Color
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.uix.slider import Slider
import matplotlib.pyplot as plt
import numpy as np
from class_percolacio_quadrat import ClassPercolacioQuadrat


##########################################################################################################################
##########################################################################################################################


#És la classe principal de l'aplicació i hereda de App (classe base per a totes les aplicacions
#en Kivy). Aquesta classe s'encarrega de configurar i gestionar les diferents pantalles de l'aplicació
#i de manejar la lògica principal del joc.
class ClusterApp(App):
    # classe constructora on tenim els links a les diferents pantalles de l'aplicació
    def build(self):
        self.screen_manager = ScreenManager()

        # Menú principal
        self.menu_principal = MenuPrincipal()
        pantalla_menu = Screen(name='menu')
        pantalla_menu.add_widget(self.menu_principal)
        self.screen_manager.add_widget(pantalla_menu)

        #Joc
        pantalla_joc = Screen(name='joc')
        pantalla_joc.add_widget(self.build_game_screen())
        self.screen_manager.add_widget(pantalla_joc)

        #Tutorial
        self.pantalla_tutorial = PantallaTutorial()
        #self.pantalla_tutorial.reset()
        pantalla_tutorial = Screen(name='tutorial')
        pantalla_tutorial.add_widget(self.pantalla_tutorial)
        self.screen_manager.add_widget(pantalla_tutorial)

        #controlar canvi de pantalla
        self.screen_manager.bind(current=self.canvi_pantalla)

        #self.pantalla_tutorial.reset()

        return self.screen_manager

##########################################################################################################################

    #construim la pantalla del joc amb ClusterWidget (classe que en permet
    #controlar les interaccions amb les cel·les del quadrat) i els controls (botons)
    def build_game_screen(self):
        main_layout = BoxLayout(orientation='horizontal')


        #part esquerra de la pantalla: és on situarem la matriu
        #actuen segons la classe ClusterWidget()
        left_layout = BoxLayout(orientation='vertical')
        self.cluster_widget = ClusterWidget()
        left_layout.add_widget(self.cluster_widget)
        main_layout.add_widget(left_layout)

        #els controls aniran a la parte de la dreta
        controls_right = BoxLayout(size_hint_x=None, width='200dp', orientation='vertical')
        generate_button = Button(text='Generate', size_hint=(1, 0.4))
        check_button = Button(text='Check Percolation', size_hint=(1, 0.4))
        self.timer_label = Label(text='Temps restant: 60 s', size_hint_y=None, height='50dp')
        self.punt_label = Label(text='Punts: 0', size_hint_y=None, height='50dp')
        generate_button.bind(on_press=self.generate_clusters)
        check_button.bind(on_press=self.check_percolation)

        #posem el botó de tornar al menú principal a la part superior dreta per evitar
        #que es clicqui sense volver mentres es juga.
        back_to_menu_button = Button(text='Tornar al menú', size_hint=(1, 0.1))
        back_to_menu_button.bind(on_press=self.tornar_menu)

        #de dalt a baix per ordre d'aparició
        controls_right.add_widget(back_to_menu_button)
        controls_right.add_widget(self.timer_label)
        controls_right.add_widget(self.punt_label)
        controls_right.add_widget(generate_button)
        controls_right.add_widget(check_button)
        main_layout.add_widget(controls_right)

        #per saber si és la primera vegada que accedim al joc, d'aquesta manera
        #es crida a generate cluster sense haver de tocar el botó generate
        #fent això també evitem que l'aplicació peti quan li donem a check percolation
        #i no hi ha cap cluster generat (ara sempre tindrem un cluster en acció).
        self.first_time_joc = True

        return main_layout

##########################################################################################################################


    def generate_clusters(self, instance=None):
        try:
            n = 10
            p = 0.5
            self.simulation = ClassPercolacioQuadrat(n, p)
            self.cluster_widget.simulation = self.simulation
            self.cluster_widget.percolated_cluster = None
            self.cluster_widget.update_canvas()
            self.cluster_widget.afegir_botons(n)
            self.cluster_widget.restar_punts(3)
            if not hasattr(self, 'time_left'):
                self.start_timer()
        except ValueError:
            print("Entrada invàlida")


##########################################################################################################################

    #funció enllaçada al botó check percolation. Serveix per veure si el sistema
    #ha percolat o no i actuar en conseqüència
    def check_percolation(self, instance):
        if self.simulation:
            perc_n = self.simulation.percola()

            if perc_n != -1:
                message = 'Ha percolat :)'
                self.cluster_widget.afegir_punts(100)  # sumem 100 per sistema percolat
                self.cluster_widget.sumar_temps(3)  # sumem 5 segons per sistema percolat
                # calculem la mida del cluster i sumem punts en base al ratio
                mida_cluster = self.simulation.mida_cluster(perc_n)
                ratio = mida_cluster / (self.simulation.n ** 2)  # de moment el denominador val 100 pq n = 10 però podria canviar
                self.cluster_widget.afegir_punts(int(50 * ratio))  # sumar els punts en base al ratio
                # Actualitzem el canvas per pintar de blanc el clsuter percolat
                self.cluster_widget.percolated_cluster = perc_n
                self.cluster_widget.update_canvas()
                # Mostrem el missatge per pantalla
                popup = Popup(title='Percolació', content=Label(text=message), size_hint=(None, None), size=(400, 200))
                popup.open()
                popup.bind(on_dismiss=lambda *args: self.generate_clusters(instance))

            else:
                message = 'No ha percolat :('
                self.cluster_widget.restar_punts(5)
                self.cluster_widget.restar_temps(5)
                popup = Popup(title='Percolació', content=Label(text=message), size_hint=(None, None), size=(400, 200))
                popup.open()
                popup.bind(on_dismiss=lambda *args: self.generate_clusters(instance))

        else:
            popup = Popup(title='Error', content=Label(text='Has de generar una matriu abans!'), size_hint=(None, None),
                          size=(400, 200))
            popup.open()

##########################################################################################################################

    #botó per tornar al menú principal
    def tornar_menu(self, instance):
        self.screen_manager.current = 'menu'


##########################################################################################################################

    #funció per iniciar el joc a 60 segons
    def start_timer(self):
        self.time_left = 60
        Clock.schedule_interval(self.update_timer, 1)


    #funció per fer que el temps passi. Es van restant segons del comptador fins
    #arribar a zero. És quan s'acaba el joc
    def update_timer(self, dt):
        self.time_left -= 1
        self.timer_label.text = f'Temps restant: {self.time_left} s'
        if self.time_left <= 0:
            self.end_game()

##########################################################################################################################


    #funció per quan s'acabi el temps. Mostrar missatge game over juntament
    #amb els punts obtinguts
    def end_game(self):
        popup = Popup(title='Game Over', content=Label(text='Temps esgotat!'), size_hint=(None, None), size=(400, 200))
        popup.open()
        self.screen_manager.current = 'menu'
        self.cluster_widget.reset_punts()
        self.cluster_widget.reset_scenario()
        Clock.unschedule(self.update_timer)


    #Aquesta funció serveix per controlar el temps i els punts quan es realitza un canvi de pantalla
    #quan es passa del joc al menú principal el que fem és atura el rellotge. Passava que de vegades el rellotge
    #seguia actiu tot i estar al menú i de sobte apareixia un missatge de temps acabat. No tenia sentit
    #quan es passa del menú principal al joc el temps es reinicia a 60 i els punts a 0.
    def canvi_pantalla(self,*args):
        #primer de tot ens ubiquem
        current_screen = self.screen_manager.current

        #estem al menú -> parem el temps
        if current_screen == 'menu':
            #parem el temps
            Clock.unschedule(self.update_timer)

        #tornem al joc -> posar els punts a 0 i el rellotge a 60
        elif current_screen == 'joc':
            self.time_left = 60
            self.timer_label.text = f'Temps restant: {self.time_left} s'
            self.cluster_widget.reset_punts()
            self.punt_label.text = 'Punts: 0'
            self.start_timer()

            if self.first_time_joc:
                Clock.schedule_once(lambda dt: self.generate_clusters(), 0.1)
                self.first_time_joc = False


        #si anem al tutorial volem que el gràfic no tingui punts a l'inici
        elif current_screen == 'tutorial':
            TutoWidget().simulation = None
            TutoWidget().percolated_cluster = None
            TutoWidget().update_canvas()



##########################################################################################################################
##########################################################################################################################


#Aquesta classe conté les propietats de BoxLayout i representa la pantalla del tutorial a l'aplicació
#Permet a l'usuari interactuar amb la gràfica i amb la matriu de clusters
class PantallaTutorial(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        top_layout = BoxLayout(size_hint_y=None, height='50dp', orientation='horizontal', padding=[10, 10, 10, 0])
        top_layout.add_widget(Widget())
        boto_menu = Button(text='Tornar', size_hint=(None, None), size=(150, 50))
        boto_menu.bind(on_press=self.tornar_menu)
        top_layout.add_widget(boto_menu)

        self.orientation = 'vertical'
        #per guardar els valors x y que es van afegint a la gràfica
        self.x_values = []
        self.y_values = []

        #per saber quins punts han percolat i pintar-los blaus o vermells (funció generate_plot)
        self.percolated_states = []

        self.simulation = ClassPercolacioQuadrat(100,0.3)

        #el main layout serà horitzontal i el dividirem en dos
        #a la dreta tindrem la gràfica i a l'esquerran la representació visual de la matriu
        main_layout = BoxLayout(orientation='horizontal')

        #configuració de la part dreta (gràfic)
        left_layout = BoxLayout(orientation='vertical')

        self.cluster_widget = TutoWidget()
        left_layout.add_widget(Label(text=''))  #potser eliminar
        left_layout.add_widget(self.cluster_widget)

        #afegim la configuració de l'esquerra al main layout
        main_layout.add_widget(left_layout)

        right_layout = BoxLayout(orientation='vertical')

        self.image = Image()
        right_layout.add_widget(self.image)

        main_layout.add_widget(right_layout)

        #afegim el main layout a la part superior del layout principal
        self.add_widget(top_layout)
        self.add_widget(main_layout)

        #layout pels controls
        #sintaxi padding: [padding_left, padding_top, padding_right, padding_bottom]
        controls_layout = BoxLayout(size_hint_y=None, height='100dp', orientation='horizontal', padding=[10, 10, 10, 50])
        self.n_input = 100
        self.p_input = Slider(min=0, max=1, value=0.3, step=0.01, size_hint=(0.6, 1))
        #cada vegada que s'interactui amb l'slider es cirarà aquesta funció per actualitzar el valor numèric que es mostra
        self.p_input.bind(value=self.on_slider_value_change)
        #mostrem el valor de la probabilirar al layout dels controls
        self.prob_label = Label(text=f'Probabilitat: {self.p_input.value:.2f}', size_hint=(0.2, 1))
        controls_layout.add_widget(self.prob_label)
        #generate_button = Button(text='Generate', size_hint=(0.2, 1))
        #generate_button.bind(on_press=self.generate_clusters)
        phase_transition_button = Button(text='Afegir punt', size_hint=(0.2, 1))
        phase_transition_button.bind(on_press=self.phase_transition)

        #controls_layout.add_widget(Label(text='Probability:', size_hint=(0.2, 1)))
        controls_layout.add_widget(self.p_input)
        #controls_layout.add_widget(generate_button)
        controls_layout.add_widget(phase_transition_button)

        self.add_widget(controls_layout)

        #generem el primer punt (d'aquesta manera ens estalviem que la part de l'esquerra sigui blanca)
        self.p_input.value = 0.3
        self.phase_transition(None)
        self.generate_clusters(None)

        #així s'actualitza la matriu dels clusters a mesure que ens anem movent per la barra de probabilitats
        self.p_input.bind(on_touch_up=self.update_clusters_on_slider_move)


    def reset(self):
        self.x_values = [0.3]
        self.y_values = [0]
        self.percolated_states = []
        self.p_input.value = 0.3
        self.simulation = ClassPercolacioQuadrat(100, 0.3)
        self.cluster_widget.simulation = self.simulation
        self.cluster_widget.percolated_cluster = None
        self.cluster_widget.update_canvas()
        self.generate_plot('phase_transition.png')
        self.image.source = 'phase_transition.png'
        self.image.reload()


    #es crida cada vegada que ens desplacem per la barra de probabilitats. Es van actualitzant els clusters
    #per tal de mostrar el que es correspon a la probabiltat del moment
    def update_clusters_on_slider_move(self, instance, touch):
        if instance.collide_point(*touch.pos):
            self.generate_clusters(None)


##########################################################################################################################


    def tornar_menu(self, instance):
        self.reset()
        App.get_running_app().root.current = 'menu'


##########################################################################################################################


    def on_slider_value_change(self, instance, value):
        #actualitzem la probabilitat quan vcanvia el valor del slider
        self.prob_label.text = f'Probability: {value:.2f}'


##########################################################################################################################


    def generate_clusters(self, instance):
        try:
            n = 100
            p = self.p_input.value
            self.simulation = ClassPercolacioQuadrat(n, p)
            self.cluster_widget.simulation = self.simulation
            self.cluster_widget.percolated_cluster = None
            self.cluster_widget.update_canvas()
        except ValueError:
            print("Entrada invàlida")


##########################################################################################################################


    #per veure si la matriu percola o no dins del tutorial
    def check_percolation(self):
        if self.simulation:
            perc_n = self.simulation.percola()
            if perc_n != -1:
                self.cluster_widget.percolated_cluster = perc_n
                self.cluster_widget.update_canvas()
                return 1
            else:
                return 0
        return None


##########################################################################################################################


    #afegim un punt a la gràfica. També mirem si el sistema percola o no.

    def phase_transition(self, instance):
        try:
            p = self.p_input.value
            self.x_values.append(p)
            #self.generate_clusters(instance)
            #Fracció del cluster més gran dins de la matriu actual
            frac = ClassPercolacioQuadrat.biggest_cluster_frac(self.simulation)
            self.y_values.append(frac)
            self.generate_plot('phase_transition.png')
            self.image.source = 'phase_transition.png'
            self.image.reload()
        except ValueError:
            print("Invalid input")


##########################################################################################################################

    #per plotejar els punts que l'usuari va introduint
    def generate_plot(self, filename):
        plt.figure()

        #verifiquem si l'últim punt ha percolat o no
        percolated = self.check_percolation()

        #inicialitzem percolated_states si no existeix
        if not hasattr(self, 'percolated_states'):
            self.percolated_states = []

        #afegim l'estat de percolació de l'últim punt
        self.percolated_states.append(percolated)

        #iterem sobre tots els punts i pintem segons el seu estat de percolació
        #vermell si ha percolat, blau si no ha percolat
        for i in range(len(self.x_values)):
            if self.percolated_states[i]:
                plt.scatter(self.x_values[i], self.y_values[i], color='red')
            else:
                plt.scatter(self.x_values[i], self.y_values[i], color='blue')

        plt.scatter([], [], color='red', label='Percola')
        plt.scatter([], [], color='blue', label='No percola')

        plt.legend(loc="upper left")
        plt.xlabel('Probabilitat')
        plt.ylabel('Fracció cluster més gran')
        plt.title('Transició de Fase')
        plt.grid(True)

        #fixem els límitts x y. Així la gràfica no es va reescalant amb cada punt que afegim
        plt.xlim(0,1)
        plt.ylim(0,1)


        plt.savefig(filename)
        plt.close()


##########################################################################################################################
##########################################################################################################################

#classe per controlar els widgets dins del tutorial (matriu)
class TutoWidget(Widget):
    def __init__(self, simulation=None, **kwargs):
        super().__init__(**kwargs)
        self.simulation = simulation
        self.percolated_cluster = None

        self.bind(size=self.update_canvas, pos=self.update_canvas)


##########################################################################################################################

    #Funció per generar un color rgb aleatori i normalitzat
    def generate_random_color(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return (r/255, g/255, b/255)


    def update_canvas(self, *args):
        self.canvas.clear()
        if self.simulation:
            with self.canvas:
                vertex_actius = self.simulation.busca_clusters()
                matriu_pintada = self.simulation.pintar_clusters(vertex_actius)
                n = len(matriu_pintada)
                cell_size = min(self.width/n, self.height/n)
                x_offset = (self.width - n*cell_size)/2
                y_offset = (self.height - n*cell_size)/2

                # Obtener todos los clusters presentes en la matriz pintada, excluyendo el primer valor (0)
                #comencem obtenint tots els clusters presents a la matriu pintada i trèiem fora el
                #primer, recordem que és un 0 i això es correspon a un vèrtex inactiu (no ens interessa ara)
                clusters = np.unique(matriu_pintada)[1:]

                # recordem que matriu pintada era una matriu o tenim tots els clusters diferenciats per números. Per exemple
                # [0,1,1]
                # [1,1,2]
                # [3,0,2]

                #Diccionari on guardaren els colors assignats als cluster. D'aquesta manera ens assegurem que cada
                #cluster (índex) té un valor assignat
                cluster_colors = {}

                #set per evitar repetir colors
                used_colors = set()

                for cluster_index in clusters:
                    #Generem colors aleatoris únics amb la funció generate_random_color
                    color = self.generate_random_color()

                    #Comprovem que aquest color no esitgui sent utilitzat
                    while color in used_colors:
                        color = self.generate_random_color()

                    #L'afegim al conjunt de colors utilitzats
                    used_colors.add(color)

                    #Finalment, assignem el color al cluster
                    cluster_colors[cluster_index] = color

                for i in range(n):
                    for j in range(n):
                        index_cluster = matriu_pintada[i, j]
                        if self.percolated_cluster is not None and matriu_pintada[i, j] == self.percolated_cluster:
                            color = (1, 1, 1)  # Blanc pel cluster que ha percolat
                        else:
                            #Obtenim el color del cluster o negre si no en té cap assignat (vèrtex inactiu)
                            color = cluster_colors.get(index_cluster, (0, 0, 0))
                        Color(*color)
                        Rectangle(pos=(x_offset + j * cell_size, Window.height/2.5 + i*cell_size), size=(cell_size, cell_size))


##########################################################################################################################
##########################################################################################################################


#classe que gestiona el menú principal
class MenuPrincipal(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.add_widget(Label(text='Menú Principal'))
        self.add_widget(Button(text='Jugar', on_press=self.iniciar_joc))
        self.add_widget(Button(text='Tutorial', on_press=self.iniciar_tutorial))


##########################################################################################################################


    # per canviar a la pantalla de joc
    def iniciar_joc(self, instance):
        App.get_running_app().root.current = 'joc'


##########################################################################################################################


    # per canviar a la pantalla del tutorial
    def iniciar_tutorial(self, instance):
        App.get_running_app().root.current = 'tutorial'


##########################################################################################################################
##########################################################################################################################


#EXPLICACIÓ ClusterWidget: és un Widget de Kivy que representa visualment la quadrícula
#on succeix la simulació de percolació
class ClusterWidget(Widget):

    # classe constructora
    def __init__(self, simulation=None, **kwargs):
        super().__init__(**kwargs)
        self.simulation = simulation
        self.percolated_cluster = None
        self.bind(size=self.update_canvas, pos=self.update_canvas)
        self.llista_botons = []
        self.punts = 0
        self.cel_actives = 0
        self.orientation = 'vertical'
        #self.n_input = TextInput(text='10', multiline=False)
        #self.p_input = TextInput(text='0.5', multiline=False)




##########################################################################################################################


    #La seva funció és dibuixar la quadrícula i els cluster segons els estats
    #de 'simulation' i 'percolated_cluster'
    def update_canvas(self, *args):
        self.canvas.clear()
        if self.simulation:
            with self.canvas:
                vertex_actius = self.simulation.busca_clusters()
                matriu_pintada = self.simulation.pintar_clusters(vertex_actius)
                n = len(matriu_pintada)
                cell_size = min(self.width/n, self.height/n)
                x_offset = (self.width - n*cell_size)/2
                y_offset = (self.height - n*cell_size)/2

                for i in range(n):
                    for j in range(n):
                        if self.percolated_cluster is not None and matriu_pintada[i, j] == self.percolated_cluster:
                            color = (0.5, random.uniform(0, 1), random.uniform(0, 1))
                        else:
                            color = (0, 0, 0) if matriu_pintada[i, j] == 0 else (1, 1, 1)
                        Color(*color)
                        Rectangle(pos=(x_offset + j*cell_size, y_offset + self.height - (i + 1)*cell_size), size=(cell_size, cell_size))


##########################################################################################################################


    #gestionar punts (sumar)
    def afegir_punts(self, punts):
        self.punts += punts
        App.get_running_app().punt_label.text = f'Punts: {self.punts}'

    #gestionar punts (restar)
    def restar_punts(self, punts):
        self.punts -= punts
        if self.punts < 0:
            self.punts = 0
        App.get_running_app().punt_label.text = f'Punts: {self.punts}'

    # gestionar punts (resetejar, iniciem a 0)
    def reset_punts(self):
        self.punts = 0
        App.get_running_app().punt_label.text = f'Punts: {self.punts}'


##########################################################################################################################


    #gestionar temps (sumar segons)
    def sumar_temps(self, segons):
        if hasattr(App.get_running_app(), 'time_left'):
            App.get_running_app().time_left += segons
            App.get_running_app().timer_label.text = f'Temps restant: {App.get_running_app().time_left}'


    #gestionar temps (restar segons)
    def restar_temps(self, segons):
        if hasattr(App.get_running_app(), 'time_left'):
            if App.get_running_app().time_left - segons < 0:
                App.get_running_app().time_left = 0
            else:
                App.get_running_app().time_left -= segons
            App.get_running_app().timer_label.text = f'Temps restant: {App.get_running_app().time_left}'


##########################################################################################################################


    #reiniciar l'escenari
    def reset_scenario(self):
        self.punts = 0
        self.percolated_cluster = None
        self.llista_botons.clear()
        self.clear_widgets()


##########################################################################################################################


    #afegim els botons (cel·les de la matriu) a la quadrícula. Aquestes cel·les tenen una
    #mida nxn. Es porta un control de les cel·les actives mitjançant la llista llista_botons
    def afegir_botons(self, n):
        self.clear_widgets()
        self.llista_botons.clear()
        cell_size = min(self.width / n, self.height / n)
        x_offset = (self.width - n * cell_size) / 2
        y_offset = (self.height - n * cell_size) / 2

        for i in range(n):
            for j in range(n):
                boto = Button(text=f"({i}, {j})", size=(cell_size, cell_size),
                              pos=(x_offset + j * cell_size, y_offset + self.height - (i + 1) * cell_size),
                              opacity=0)
                boto.row = i
                boto.col = j
                boto.bind(on_press=self.boto_clicat)

                self.llista_botons.append(boto)
                self.add_widget(boto)


##########################################################################################################################


    #Actu1a en conseqüència que es fa clic en un botó de la quadrícula.
    #És a dir, comprova si la cel·la està ocupada i en cas de no estar-ho
    #l'afegeix i restem 3 punts si no és la 1a vegada. Altrament no fa res.
    def boto_clicat(self, instance):
        if self.simulation.matriu[instance.row, instance.col] == 0:
            self.simulation.matriu[instance.row, instance.col] = 1
            instance.background_color = (1, 1, 1, 1)
            self.update_canvas()
            self.cel_actives += 1
            if self.cel_actives > 1:
                self.restar_punts(3)


##########################################################################################################################
##########################################################################################################################


if __name__ == '__main__':
    ClusterApp().run()








