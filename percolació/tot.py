import random
import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Color
from kivy.uix.popup import Popup
from class_percolacio_quadrat import ClassPercolacioQuadrat
from kivy.clock import Clock
import percotuto
from kivy.uix.behaviors import ButtonBehavior


#EN CONTRUCCIÓ!


##########################################################################################################################
##########################################################################################################################

                                                #CLASSE AMB COSES DEL JOC I DEL TUTORIAL
class ClusterWidget(Widget):
    # Mètode constructor de la classe

    #funció tant del tutorial com del joc
    def __init__(self, simulation=None, **kwargs):
        super().__init__(**kwargs)
        self.simulation = simulation
        # Amb aquesta variable identificarem el cluster que ha percolat i el pintarem de blanc
        self.percolated_cluster = None
        self.bind(size=self.update_canvas, pos=self.update_canvas)
        self.llista_botons = []
        # sistema de punts
        self.punts = 0
        # comptador de cel·les activades pel jugador (a partir de la 2a restem punts)
        self.cel_actives = 0
        self.orientation = 'vertical'


    def update_canvas(self, *args):
        self.canvas.clear()
        if self.simulation:
            with self.canvas:
                vertex_actius = self.simulation.busca_clusters()
                matriu_pintada = self.simulation.pintar_clusters(vertex_actius)
                n = len(matriu_pintada)
                cell_size = min(self.width / n, self.height / n)
                x_offset = (self.width - n * cell_size) / 2
                y_offset = (self.height - n * cell_size) / 2

                for i in range(n):
                    for j in range(n):
                        if self.percolated_cluster is not None and matriu_pintada[i, j] == self.percolated_cluster:
                            color = (1, 1, 1)
                        else:
                            color = (0, 0, 0) if matriu_pintada[i, j] == 0 else (matriu_pintada[i, j] / len(vertex_actius), 0, 1)
                        Color(*color)
                        Rectangle(pos=(x_offset + j * cell_size, y_offset + self.height - (i + 1) * cell_size), size=(cell_size, cell_size))


    #funcions del joc
    def afegir_punts(self, punts):
        self.punts += punts
        # mostrem els punts per pantalla:
        App.get_running_app().punt_label.text = f'Punts: {self.punts}'


    def reset_scenario(self):
        self.punts = 0
        self.percolated_cluster = None
        self.llista_botons.clear()
        self.clear_widgets()


    def restar_punts(self, punts):
        self.punts -= punts
        # no permetem punts negatius
        if self.punts < 0:
            self.punts = 0
        # mostrar els punts per pantalla
        App.get_running_app().punt_label.text = f'Punts: {self.punts}'


    def reset_punts(self):
        self.punts = 0
        # mostrar els punts per pantalla
        App.get_running_app().punt_label.text = f'Punts: {self.punts}'


    def sumar_temps(self, segons):
        if hasattr(App.get_running_app(), 'time_left'):
            App.get_running_app().time_left += segons
            App.get_running_app().timer_label.text = f'Temps restant: {App.get_running_app().time_left}'


    def restar_temps(self, segons):
        if hasattr(App.get_running_app(), 'time_left'):
            # no volem temps negatius
            if App.get_running_app().time_left - segons < 0:
                App.get_running_app().time_left = 0
            else:
                App.get_running_app().time_left -= segons
            App.get_running_app().timer_label.text = f'Temps restant: {App.get_running_app().time_left}'


    def afegir_botons(self, n):

        # borrón y cuenta nueva
        self.clear_widgets()
        self.llista_botons.clear()
        for botons in self.llista_botons:
            self.remove_widget(botons)

        # ara afegim els botons a cada cel·la de la quadrícula
        cell_size = min(self.width / n, self.height / n)
        x_offset = (self.width - n * cell_size) / 2
        y_offset = (self.height - n * cell_size) / 2

        for i in range(n):
            for j in range(n):
                boto = Button(text=f"({i}, {j})", size=(cell_size, cell_size),
                              pos=(x_offset + j * cell_size, y_offset + self.height - (i + 1) * cell_size), opacity=0)
                # associem una posició de la matriu al seu botó corresponent
                boto.row = i
                boto.col = j
                boto.bind(on_press=self.boto_clicat)

                # quan el punter passi per sobre d'una cel·la aquesta canvia lleugerament el seu color (així saps on et trobes)
                # boto.bind(on_enter=self.on_mouse_enter) #en contrucción
                # boto.bind(on_leave=self.on_mouse_leave)

                self.llista_botons.append(boto)
                self.add_widget(boto)


    def boto_clicat(self, instance):
        print(f"Botón presionado en posición: ({instance.row}, {instance.col})")
        # print('botó clicat')
        if self.simulation.matriu[instance.row, instance.col] == 1:
            print("Blanc")

        # canviem el color i comptem a cel·la com activa
        elif self.simulation.matriu[instance.row, instance.col] == 0:
            self.simulation.matriu[instance.row, instance.col] = 1  # la marquem com ocupada
            instance.background_color = (1, 1, 1, 1)  # canviem el seu color a blanc
            self.update_canvas()  # actualitzem el canvas per fer visible el canvi
            print("negre")
            self.cel_actives += 1
            if self.cel_actives > 1:
                # comencem a restar per cel·la addicional (poster canvio el llindar)
                self.restar_punts(3)


##########################################################################################################################
##########################################################################################################################
                                            #SCREEN MANAGER

#Des d'aquí decidim si anar al joc o al tutorial


class ClusterApp(App):
    def build(self):

        self.screen_manager = ScreenManager()

        self.menu_principal = MenuPrincipal()
        pantalla_menu = Screen(name='menu')
        pantalla_menu.add_widget(self.menu_principal)
        self.screen_manager.add_widget(pantalla_menu)

        self.pantalla_tutorial = PantallaTutorial()
        pantalla_tutorial = Screen(name='tutorial')
        pantalla_tutorial.add_widget(self.pantalla_tutorial)
        self.screen_manager.add_widget(pantalla_tutorial)

        pantalla_joc = Screen(name='joc')
        pantalla_joc.add_widget(self.build_game_screen())
        self.screen_manager.add_widget(pantalla_joc)

        return self.screen_manager

    ##########################################################################################################################

    def build_game_screen(self):
        main_layout = BoxLayout(orientation='horizontal')

        left_layout = BoxLayout(orientation='vertical')
        self.cluster_widget = ClusterWidget()
        left_layout.add_widget(self.cluster_widget)
        main_layout.add_widget(left_layout)

        controls_right = BoxLayout(size_hint_x=None, width='200dp', orientation='vertical')
        generate_button = Button(text='Generate', size_hint=(1, 0.4))
        check_button = Button(text='Check Percolation', size_hint=(1, 0.4))
        self.timer_label = Label(text='Temps restant: 60 s', size_hint_y=None, height='50dp')
        self.punt_label = Label(text='Punts: 0', size_hint_y=None, height='50dp')
        generate_button.bind(on_press=self.generate_clusters)
        check_button.bind(on_press=self.check_percolation)

        # botó per tornar al menú principal des del joc:
        back_to_menu_button = Button(text='Tornar al menú', size_hint=(1, 0.2))
        back_to_menu_button.bind(on_press=self.tornar_menu)

        controls_right.add_widget(self.timer_label)
        controls_right.add_widget(self.punt_label)
        controls_right.add_widget(generate_button)
        controls_right.add_widget(check_button)
        controls_right.add_widget(back_to_menu_button)
        main_layout.add_widget(controls_right)

        return main_layout

    ##########################################################################################################################

    def generate_clusters(self, instance):
        # self.cluster_widget.reset_scenario() #borrón y cuenta nueva
        n = 10
        p = 0.5
        self.simulation = ClassPercolacioQuadrat(n, p)
        self.cluster_widget.simulation = self.simulation
        # inicialitzem l'index del cluster percolat a None (si no hem comprovat la percolació
        # no podem saber si ha percolat o no)
        self.cluster_widget.percolated_cluster = None
        self.cluster_widget.update_canvas()
        # afegim els botons per fer les cel·les interactives
        self.cluster_widget.afegir_botons(n)
        # resem per generar un escenari nou
        self.cluster_widget.restar_punts(3)
        # self.cluster_widget.cel_actives = 0
        # inicialitzem el temps si encara no estava en marxa
        if not hasattr(self, 'time_left'):
            self.start_timer()

    ##########################################################################################################################

    def check_percolation(self, instance):
        if self.simulation:
            perc_n = self.simulation.percola()

            if perc_n != -1:
                message = 'Ha percolat :)'
                self.cluster_widget.afegir_punts(100)  # sumem 100 per sistema percolat
                self.cluster_widget.sumar_temps(3)  # sumem 5 segons per sistema percolat
                # calculem la mida del cluster i sumem punts en base al ratio
                mida_cluster = self.simulation.mida_cluster(perc_n)
                ratio = mida_cluster / (
                        self.simulation.n ** 2)  # de moment el denominador val 100 pq n = 10 però podria canviar
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

    # Inicializtem el rellotge en 60 segons
    def start_timer(self):
        # temps inicial = 60 s
        self.time_left = 60
        # això assegura que el temporitzador s'actualitzi inmediatament a l'iniciar el joc
        self.update_timer(0)
        # Així es crida la funció update_timer cada segon
        Clock.schedule_interval(self.update_timer, 1)

    ##########################################################################################################################

    # comptador de temps
    def update_timer(self, dt):
        # encara tenim temps -> restem segons fins que s'acabi
        if self.time_left > 0:
            self.time_left -= 1
            self.timer_label.text = f'Temps restant: {self.time_left}'
        # s'ha acabat el temps :(
        else:
            Clock.unschedule(self.update_timer)
            self.timeup_popup()

    ##########################################################################################################################

    # Funció que es crida quan s'acaba el temps.
    def timeup_popup(self):
        popup = Popup(title='TEMPS ESGOTAT', content=Label(
            text=f'Punts obtinguts: {self.cluster_widget.punts}\nToca la pantalla per seguir jugant'),
                      size_hint=(None, None), size=(400, 200))
        popup.open()
        popup.bind(on_dismiss=self.reset_and_generate_clusters)

    def reset_and_generate_clusters(self, instance):
        self.cluster_widget.reset_scenario()
        self.generate_clusters(None)
        self.start_timer()

    def tornar_menu(self, instance):
        self.screen_manager.current = 'menu'


##########################################################################################################################
##########################################################################################################################
                                                        #MENÚ PRINCIPAL


class MenuPrincipal(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.add_widget(Label(text='Menú Principal'))
        self.add_widget(Button(text='Jugar', on_press=self.iniciar_joc))
        self.add_widget(Button(text='Tutorial', on_press=self.iniciar_tutorial))

    def iniciar_joc(self,instance):
        App.get_running_app().root.current = 'joc'


    def iniciar_tutorial(self,instance):
        App.get_running_app().root.current = 'tutorial'


##########################################################################################################################
##########################################################################################################################

                                            # TUTORIAL
class PantallaTutorial(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.add_widget(Label(text='Contingut tutorial (en construcció)'))
        self.add_widget(Button(text='Tornar al Menú', on_press=self.tornar_menu))

    def tornar_menu(self, instance):
        App.get_running_app().root.current = 'menu'


if __name__ == '__main__':
    ClusterApp().run()
