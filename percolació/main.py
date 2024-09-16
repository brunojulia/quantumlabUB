import random
import kivy
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Rectangle, Color
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.uix.slider import Slider
from kivy.lang import Builder
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

        # per generar el .kv
        #Builder.load_file('cluster.kv')
        #per tenir sempre la pantalla completa i que no es generir errors de reescalat
        Window.fullscreen = True
        Window.size = (1920,1080)
        Window.borderless = False

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
        pantalla_tutorial = Screen(name='tutorial')
        pantalla_tutorial.add_widget(self.pantalla_tutorial)
        self.screen_manager.add_widget(pantalla_tutorial)

        #controlar canvi de pantalla
        self.screen_manager.bind(current=self.canvi_pantalla)

        #fem resize de la pantalla per no perdre les proporcions
        #Window.bind(on_resize=self.ajustar_proporcions)


        return self.screen_manager

##########################################################################################################################


    #construim la pantalla del joc amb ClusterWidget (classe que en permet
    #controlar les interaccions amb les cel·les del quadrat) i els controls (botons)
    def build_game_screen(self):


        main_layout = BoxLayout(orientation='horizontal',padding=0, spacing=0)


        #part esquerra de la pantalla: és on situarem la matriu
        #actuen segons la classe ClusterWidget()
        left_layout = BoxLayout(orientation='vertical')
        self.cluster_widget = ClusterWidget()
        left_layout.add_widget(self.cluster_widget)
        main_layout.add_widget(left_layout)

        #els controls aniran a la parte de la dreta
        controls_right = BoxLayout(size_hint_x=None, width='600dp', orientation='vertical')
        generate_button = Button(text='Generate', size_hint=(1, 0.1), font_name='atari',font_size='40sp',background_normal='boto2.png',padding=[0, 0, 0, 70],color=(0, 0, 0, 1))
        check_button = Button(text='Check Percolation', size_hint=(1, 0.15), font_name='atari',font_size='40sp',background_normal='boto.png',padding=[0, 0, 0, 130],color=(0, 0, 0, 1))
        self.timer_label = Label(text='Temps restant: 60 s', size_hint_y=None, height='50dp', font_name='atari',font_size='40sp')
        self.punt_label = Label(text='Punts: 0', size_hint_y=None, height='50dp', font_name='atari',font_size='40sp')
        generate_button.bind(on_press=self.generate_clusters)
        check_button.bind(on_press=self.check_percolation)

        #posem el botó de tornar al menú principal a la part superior dreta per evitar
        #que es clicqui sense volver mentres es juga.
        back_to_menu_button = Button(text='Tornar al menú', size_hint=(0.5, 0.05), font_name='atari',font_size='30sp',background_normal='boto3.png',padding=[0, 0, 0, 40],color=(0, 0, 0, 1),pos_hint={'right': 0.75})
        back_to_menu_button.bind(on_press=self.tornar_menu)

        #de dalt a baix per ordre d'aparició
        controls_right.add_widget(back_to_menu_button)
        controls_right.add_widget(self.timer_label)
        controls_right.add_widget(self.punt_label)
        controls_right.add_widget(check_button)
        controls_right.add_widget(generate_button)
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
            n = 13
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
                self.cluster_widget.sumar_temps(3)  # sumem 3 segons per sistema percolat
                mida_cluster = self.simulation.mida_cluster(perc_n) #mida del cluster que ha percolat
                self.cluster_widget.afegir_punts(100 + mida_cluster)  # sumar els punts en base la mida del cluster percolat

                # Actualitzem el canvas per pintar de blanc el clsuter percolat
                message = f'+ {100 + mida_cluster} punts \n + 3 segons \n \n Toca per continuar'
                self.cluster_widget.percolated_cluster = perc_n
                self.cluster_widget.update_canvas()

                # Mostrem el missatge per pantalla amb estètica retro
                layout = BoxLayout(orientation='vertical',padding=10,spacing=10)

                message_label = Label(text=message, font_name='atari',font_size='24sp',color=(1,1,1,1), halign='center', valign='middle')
                layout.add_widget(message_label)

                #popup = Popup(title='Percolació', content=Label(text=message), size_hint=(None, None), size=(400, 200))
                popup = Popup(
                    title='Ha percolat :)',
                    title_font='atari',
                    content=layout,
                    size_hint=(None,None),
                    size=(500,300),
                    background='black.png',
                    separator_color=[0,0,0,0]
                )
                popup.open()
                popup.bind(on_dismiss=lambda *args: self.generate_clusters(instance))

            else:

                self.cluster_widget.restar_punts(5)
                self.cluster_widget.restar_temps(5)

                message = f'- 5 punts \n - 5 segons \n \n Toca per continuar'
                # Mostrem el missatge per pantalla amb estètica retro
                layout = BoxLayout(orientation='vertical',padding=10,spacing=10)

                message_label = Label(text=message, font_name='atari',font_size='24sp',color=(1,1,1,1), halign='center', valign='middle')
                layout.add_widget(message_label)

                #popup = Popup(title='Percolació', content=Label(text=message), size_hint=(None, None), size=(400, 200))
                popup = Popup(
                    title='No ha percolat :(',
                    title_font='atari',
                    content=layout,
                    size_hint=(None,None),
                    size=(500,300),
                    background='black.png',
                    separator_color=[0,0,0,0]
                )
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

        # Mostrem el missatge per pantalla amb estètica retro
        message = f'Temps esgotat \n \n {self.cluster_widget.punts} punts'
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        message_label = Label(text=message, font_name='atari', font_size='50sp', color=(1, 1, 1, 1), halign='center', valign='middle')
        layout.add_widget(message_label)

        # popup = Popup(title='Percolació', content=Label(text=message), size_hint=(None, None), size=(400, 200))
        popup = Popup(
            title='',
            title_font='atari',
            content=layout,
            size_hint=(None, None),
            size=(600, 400),
            background='black.png',
            separator_color=[0, 0, 0, 0]
        )
        popup.bind(on_touch_down=lambda instance, touch: self.canvi_pantalla('joc'))
        popup.open()
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
            self.generate_clusters()
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

        #missatges tutorial
        self.step = 0

        #fons de pantalla
        self.background = Image(source='fons_menu.png', allow_stretch=False, keep_ratio=True)
        self.add_widget(self.background)

        self.top_layout = BoxLayout(size_hint_y=None, height='50dp', orientation='horizontal', padding=[10, 10, 30, 390])
        self.top_layout.add_widget(Widget())
        self.boto_menu = Button(text='Tornar', size_hint=(None, None), size=(150, 100), background_normal='boto3.png',halign='center', valign='middle', font_name='atari', font_size='20sp')
        self.boto_menu.bind(on_press=self.tornar_menu)
        self.top_layout.add_widget(self.boto_menu)

        self.orientation = 'vertical'
        #per guardar els valors x y que es van afegint a la gràfica
        self.x_values = []
        self.y_values = []

        #per saber quins punts han percolat i pintar-los blaus o vermells (funció generate_plot)
        self.percolated_states = []

        self.simulation = ClassPercolacioQuadrat(100,0.3)

        #el main layout serà horitzontal i el dividirem en dos
        #a la dreta tindrem la gràfica i a l'esquerran la representació visual de la matriu
        self.main_layout = BoxLayout(orientation='horizontal',size_hint=(1,1))

        #configuració de la part esquerra (clusters)
        self.left_layout = FloatLayout(size_hint=(0.6, 1))
        self.cluster_widget = TutoWidget(size_hint=(None,None),size=(Window.width * 0.6, 600))
        self.left_layout.add_widget(self.cluster_widget)
        #afegim la configuració de l'esquerra al main layout
        self.main_layout.add_widget(self.left_layout)

        #Configuració part dreta (gràfic)
        self.right_layout = FloatLayout(size_hint=(0.4, 1))
        self.image = Image(size_hint=(None, None), size=(Window.width * 0.4, 600), allow_stretch=True, keep_ratio=True)
        self.image.pos_hint = {'center_x': 0.3, 'top': 1.7} #centrem a la part dreta
        self.right_layout.add_widget(self.image)
        self.main_layout.add_widget(self.right_layout)

        #afegim el main layout a la part superior del layout principal
        self.add_widget(self.top_layout)
        self.add_widget(self.main_layout)

        #layout pels controls
        #sintaxi padding: [padding_left, padding_top, padding_right, padding_bottom]
        self.controls_layout = BoxLayout(size_hint_y=None, height='100dp', orientation='horizontal', padding=[10, 10, 10, 50])
        self.n_input = 100
        self.p_input = Slider(min=0, max=1, value=0.3, step=0.01, size_hint=(0.6, 1))
        #cada vegada que s'interactui amb l'slider es cirarà aquesta funció per actualitzar el valor numèric que es mostra
        self.p_input.bind(value=self.on_slider_value_change)
        #mostrem el valor de la probabilirar al layout dels controls
        self.prob_label = Label(text=f'Probabilitat: {self.p_input.value:.2f}', size_hint=(0.2, 1))
        self.controls_layout.add_widget(self.prob_label)
        #generate_button = Button(text='Generate', size_hint=(0.2, 1))
        #generate_button.bind(on_press=self.generate_clusters)
        self.phase_transition_button = Button(text='Afegir punt', size_hint=(None, None), background_normal = 'boto.png',size=(300,140),halign='center', valign='middle', font_name='atari', font_size='30sp')
        self.phase_transition_button.bind(on_press=self.phase_transition)

        #controls_layout.add_widget(Label(text='Probability:', size_hint=(0.2, 1)))
        self.controls_layout.add_widget(self.p_input)
        #controls_layout.add_widget(generate_button)
        self.controls_layout.add_widget(self.phase_transition_button)

        self.add_widget(self.controls_layout)

        #generem el primer punt (d'aquesta manera ens estalviem que la part de l'esquerra sigui blanca)
        self.p_input.value = 0.3
        self.phase_transition(None)
        self.generate_clusters(None)

        #així s'actualitza la matriu dels clusters a mesure que ens anem movent per la barra de probabilitats
        self.p_input.bind(on_touch_up=self.update_clusters_on_slider_move)

        self.show_step()

    #per tenir més control sobre l'estètica de cada missatge en el tutorial
    def show_message(self,message,callback=None,font_name='atari',font_size='20sp'):

        self.clear_widgets()
        message_label = Label(text=message, size_hint=(1, 0.8), font_name=font_name, font_size=font_size,halign='center', valign='middle')
        self.add_widget(message_label)


    def show_step(self):

        self.clear_widgets()
        callback=None

        if self.step == 0:
            self.show_message('Què és la percolació?', self.next_step, font_name='atari',font_size='80sp')

            #botó
            next_button = Button(text='Next', size_hint=(0.1, 0.1), background_normal='boto3.png', font_name='atari',font_size='30sp')
            next_button.bind(on_press=lambda x: (self.next_step()))
            self.add_widget(next_button)

        elif self.step == 1:

            main_layout = BoxLayout(orientation='horizontal', size_hint=(1, 1))

            #layout de l'esquerra. Es divideix en dues parts: la superior (pel text) i la inferior (pel botó de next)
            left_layout = BoxLayout(orientation='vertical', size_hint=(0.4, 1))

            #layout esquerre superior (text)
            top_left_layout = BoxLayout(size_hint=(1,1))
            text_label = Label(text='Imagina que el riu vol anar de dalt a baix. \nVeiem que això ja és possible!', font_name='atari', font_size='40sp',halign='center', valign='middle')
            top_left_layout.add_widget(text_label)

            #layout esquerre inferior (botó next)
            bottom_left_layout = BoxLayout(size_hint=(0.2, 0.29))
            next_button = Button(text='Next', size_hint=(0.5, 0.5), background_normal='boto3.png', font_name='atari', font_size='30sp')
            next_button.bind(on_press=lambda x: (self.next_step()))
            bottom_left_layout.add_widget(next_button)

            left_layout.add_widget(top_left_layout)
            left_layout.add_widget(bottom_left_layout)

            #layout dret per a la imatge del riu
            right_layout = BoxLayout(size_hint=(0.35, 1))
            image = Image(source='rio.png', size_hint=(1, 1), allow_stretch=True, keep_ratio=False)
            right_layout.add_widget(image)

            main_layout.add_widget(left_layout)
            main_layout.add_widget(right_layout)

            self.add_widget(main_layout)

        elif self.step == 2:
            self.show_message('Ara imagina un escenari més complicat', self.next_step,font_name='atari',font_size='80sp')

            #botó
            next_button = Button(text='Next', size_hint=(0.1, 0.1), background_normal='boto3.png', font_name='atari',font_size='30sp')
            next_button.bind(on_press=lambda x: (self.next_step()))
            self.add_widget(next_button)

        elif self.step == 3:
            main_layout = BoxLayout(orientation='horizontal', size_hint=(1, 1))

            # Layout de la izquierda, dividido en dos partes: superior (texto) e inferior (botón)
            left_layout = BoxLayout(orientation='vertical', size_hint=(0.4, 1))

            # Layout izquierdo superior (texto)
            top_left_layout = BoxLayout(size_hint=(1, 1))
            text_label = Label(text='Què és aquesta graella amb números?',
                               font_name='atari', font_size='45sp', halign='center', valign='middle', size_hint=(1, 1))
            text_label.bind(size=text_label.setter('text_size'))
            top_left_layout.add_widget(text_label)

            # Layout izquierdo inferior (botón)
            bottom_left_layout = BoxLayout(size_hint=(0.2, 0.29))
            next_button = Button(text='Next', size_hint=(0.5, 0.5), background_normal='boto3.png', font_name='atari', font_size='30sp')
            next_button.bind(on_press=lambda x: self.next_step())
            bottom_left_layout.add_widget(next_button)

            left_layout.add_widget(top_left_layout)
            left_layout.add_widget(bottom_left_layout)

            # Layout derecho para la imagen
            right_layout = BoxLayout(size_hint=(0.4, 1))
            image = Image(source='quadrat.png', size_hint=(1, 1), allow_stretch=True, keep_ratio=True)
            right_layout.add_widget(image)

            main_layout.add_widget(left_layout)
            main_layout.add_widget(right_layout)

            # Añadir el layout principal a la pantalla
            self.add_widget(main_layout)



        elif self.step == 4:
            #layout principal
            main_layout = BoxLayout(orientation='horizontal', size_hint=(1, 1))

            # layput de l'esquerre
            left_layout = BoxLayout(orientation='vertical', size_hint=(0.5, 1))

            #text
            top_left_layout = BoxLayout(size_hint=(1, 0.6))
            message_label = Label(text='Tria un número', font_name='atari', font_size='80sp', halign='center', valign='middle', size_hint=(1, 1))
            top_left_layout.add_widget(message_label)

            #layout pels botons (part interactiva del tutorial)
            bottom_left_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.4))

            #botó de 0.8
            button_08 = Button(text='0.8', size_hint=(0.2, 0.9), font_size='70sp', background_normal='boto.png', font_name='atari',padding=[0, 20])
            button_08.bind(on_press=lambda x: self.change_step(5))
            bottom_left_layout.add_widget(button_08)

            #botó de 0.3
            button_03 = Button(text='0.3', size_hint=(0.2, 1), font_size='70sp', background_normal='boto3.png', font_name='atari')
            button_03.bind(on_press=lambda x: self.change_step(6)) #anem al 6è pas
            bottom_left_layout.add_widget(button_03)

            left_layout.add_widget(top_left_layout)
            left_layout.add_widget(bottom_left_layout)

            #layout de la dreta per a la imatge
            right_layout = BoxLayout(size_hint=(0.5, 1))
            image = Image(source='quadrat.png', size_hint=(1, 1), allow_stretch=True, keep_ratio=True)
            right_layout.add_widget(image)

            main_layout.add_widget(left_layout)
            main_layout.add_widget(right_layout)

            self.add_widget(main_layout)


            #espai inferior
            spacer = Widget(size_hint=(1, 0.6))
            left_layout.add_widget(spacer)


        elif self.step == 6:
            self.show_message('Però hi ha un valor a partir del qual el riu sempre fluirà de dalt a baix!', self.next_step,font_name='atari',font_size='30sp')

            #botó
            next_button = Button(text='Next', size_hint=(0.1, 0.1), background_normal='boto3.png', font_name='atari',font_size='30sp')
            next_button.bind(on_press=lambda x: (self.next_step()))
            self.add_widget(next_button)

        elif self.step == 7:
            self.show_message("Sabries dir quin és aquest valor?", self.end_tutorial, font_name='atari', font_size='30sp')

            #botó
            next_button = Button(text='Next', size_hint=(0.1, 0.1), background_normal='boto3.png', font_name='atari',font_size='30sp')
            next_button.bind(on_press=lambda x: (self.end_tutorial()))
            self.add_widget(next_button)


    def display_message(self, message, callback=None):
        self.clear_widgets()
        message_label = Label(text=message, size_hint=(1, 0.8))
        self.add_widget(message_label)
        next_button = Button(text='Next', size_hint=(1, 0.2))
        next_button.bind(on_press=lambda x: (callback() if callback else self.next_step()))
        self.add_widget(next_button)

    #avancem en el tutorial
    def next_step(self, instance=None):
        self.step += 1
        self.show_step()

    def change_step(self,new_step):
        self.step = new_step
        self.show_step()

    #per finalitzar el tutorial i poder jugar amb els clusters
    def end_tutorial(self):
        self.step = 0
        self.clear_widgets()
        self.add_widget(self.background)
        self.add_widget(self.top_layout)
        self.add_widget(self.main_layout)
        self.add_widget(self.controls_layout)

        #restaurem la gràfica i els controls
        self.generate_clusters(None)
        self.image.source = 'phase_transition.png'
        self.image.reload()


    def reset(self):
        self.x_values = [0.3]
        self.y_values = [0]
        self.step = 0
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
                cell_size = min(self.width/n, self.height/n)*1.4
                x_offset = (self.width - n*cell_size)/2 - 100 #per centrar en x
                y_offset = (self.height - n*cell_size)/2 + 285 #aquest +285 és per centrar la matriu amb el gràfic

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
                        Rectangle(pos=(x_offset + j * cell_size, y_offset + i * cell_size), size=(cell_size, cell_size))


##########################################################################################################################
##########################################################################################################################


#classe que gestiona el menú principal
class MenuPrincipal(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.orientation = 'vertical'
        #self.add_widget(Label(text='Menú Principal'))
        #self.add_widget(Button(text='Jugar', on_press=self.iniciar_joc))
        #self.add_widget(Button(text='Tutorial', on_press=self.iniciar_tutorial))


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

        #carreguem la imatge d'aigua pels vèrtexs actius
        img = CoreImage('agua.png')
        self.water_texture = img.texture

        #carreguem la imatge de terra pels vèrtexs inactius
        img2 = CoreImage('tierra2.png')
        self.land_texture = img2.texture

        #pedra als costats per donar a entendre que la percolació va de dalt a abaix
        img3 = CoreImage('stone.png')
        self.stone_texture = img3.texture


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

                # Dibuixar la imatge de pedra a la part esquerra i dreta
                if self.stone_texture:
                    # Imatge de pedra a l'esquerra
                    Rectangle(pos=(x_offset - cell_size, y_offset),
                              size=(cell_size, self.height),texture=self.stone_texture)
                    # Imatge de pedra a la dreta
                    Rectangle(pos=(x_offset + n * cell_size, y_offset),
                              size=(cell_size, self.height),texture=self.stone_texture)


                for i in range(n):
                    for j in range(n):
                        if self.percolated_cluster is not None and matriu_pintada[i, j] == self.percolated_cluster:
                            color = (0.5, random.uniform(0, 1), random.uniform(0, 1))
                            Rectangle(pos=(x_offset + j * cell_size, y_offset + self.height - (i + 1) * cell_size),
                                      size=(cell_size, cell_size), texture = None)

                        else:
                            texture = self.land_texture if matriu_pintada[i,j] == 0 else self.water_texture
                            Rectangle(pos=(x_offset + j * cell_size, y_offset + self.height - (i + 1) * cell_size),
                                      size=(cell_size, cell_size), texture=texture)


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


    #Actua en conseqüència que es fa clic en un botó de la quadrícula.
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
#Builder.load_file('cluster.kv')

if __name__ == '__main__':
    ClusterApp().run()









