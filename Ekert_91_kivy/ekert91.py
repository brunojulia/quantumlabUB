import time #revisar
import numpy as np

from QKD_functions import * 

from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
from kivy.core.window import Window
from kivy.clock import Clock

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label #revisar
from kivy.animation import Animation
from kivy.vector import Vector
from kivy.uix.screenmanager import ScreenManager, Screen

from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, BooleanProperty, StringProperty, Property

OFFSET = 12

class Round_Button(Button):

    def on_touch_down(self, touch):
        dist = np.sqrt( pow(touch.pos[0] - self.center[0], 2) + pow(touch.pos[1] - self.center[1], 2) )
        if dist <= 38:  # radius of the circle
            return super(Button, self).on_touch_down(touch)
        else:
            return False
        
class Switch_Button(Button):

    active = BooleanProperty(False)

    def change_mode(self):
        self.active = not self.active
        

class Photon(Widget):

    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self):
        self.pos = Vector(*self.velocity) + self.pos


class Slit(Widget):

    angle = NumericProperty(0)

    def change_left_angles(self):

        match self.angle:
            case 0:
                self.angle = 22.5
            case 22.5:
                self.angle = 45
            case 45:
                self.angle = 0

    def change_right_angles(self):

        match self.angle:
            case 0:
                self.angle = 22.5
            case 22.5:
                self.angle = -22.5
            case -22.5:
                self.angle = 0

    def angle_to_basis(self):

        match self.angle:
            case -22.5:
                return -1
            case 0:
                return 0
            case 22.5:
                return 1
            case 45:
                return 2

class MenuScreen(Screen):
    pass

# TO BE DONE (maybe)
class SettingsScreen(Screen):
        
    def catalan(self):
        self.manager.get_screen('menu').label_play_button.text = 'Jugar'
        self.manager.get_screen('menu').label_settings_button.text = 'Ajustaments'
        self.label_settings.text = 'Ajustaments'
        self.settings_menu_button.text = 'Menú'
        self.manager.get_screen('game').label_source.text = 'Font'
        self.manager.get_screen('game').label_expert_mode.text = 'Mode Expert'
        self.manager.get_screen('game').label_eavesdropping.text = 'Espionatge'
        self.manager.get_screen('game').game_menu_button.text = 'Menú'
        self.manager.get_screen('game').tut_press_source.text = "Premeu-me!"
        self.manager.get_screen('game').tut_emmited.text = "Heu emès un parell\nde fotons!"
        self.manager.get_screen('game').tut_alice_button.text = "Premeu aquí per canviar\nla base d'Alice"
        self.manager.get_screen('game').tut_bob_button.text = "I aquí per canviar\nla base d'en Bob"
        self.manager.get_screen('game').tut_alice_bit.text = "Aquests són els bits d'Alice"
        self.manager.get_screen('game').tut_bob_bit.text = "I aquests els bits d'en Bob"
        self.manager.get_screen('game').tut_eve.text = "Premeu aquí per afegir un espia"
        self.manager.get_screen('game').tut_eve_left.text = "Eve pot escoltar des de l'esquerra"
        self.manager.get_screen('game').tut_eve_right.text = "I des de la dreta"
        self.manager.get_screen('game').tut_slider.text = "Llisqueu per escollir quants\nparells envieu"
        self.manager.get_screen('game').tut_eve_expert.text = "Premeu aquí per a més detalls"

    def spanish(self):
        self.manager.get_screen('menu').label_play_button.text = 'Jugar'
        self.manager.get_screen('menu').label_settings_button.text = 'Ajustes'
        self.label_settings.text = 'Ajustes'
        self.settings_menu_button.text = 'Menú'
        self.manager.get_screen('game').label_source.text = 'Fuente'
        self.manager.get_screen('game').label_expert_mode.text = 'Modo Experto'
        self.manager.get_screen('game').label_eavesdropping.text = 'Espionaje'
        self.manager.get_screen('game').game_menu_button.text = 'Menú'
        self.manager.get_screen('game').tut_press_source.text = "Pulsadme!"
        self.manager.get_screen('game').tut_emmited.text = "Habéis emitido un par\nde fotones!"
        self.manager.get_screen('game').tut_alice_button.text = "Pulsad aquí para cambiar\nla base de Alice"
        self.manager.get_screen('game').tut_bob_button.text = "Y aquí para cambiar\nla base de Bob"
        self.manager.get_screen('game').tut_alice_bit.text = "Estos son los bits de Alice"
        self.manager.get_screen('game').tut_bob_bit.text = "Y estos los bits de Bob"
        self.manager.get_screen('game').tut_eve.text = "Pulsad aquí para añadir un espía"
        self.manager.get_screen('game').tut_eve_left.text = "Eve puede escuchar desde la izquierda"
        self.manager.get_screen('game').tut_eve_right.text = "Y desde la derecha"
        self.manager.get_screen('game').tut_slider.text = "Deslizad para escoger cuántos\npares enviáis"
        self.manager.get_screen('game').tut_eve_expert.text = "Pulsad aquí para más detalles"

    def english(self):
        self.manager.get_screen('menu').label_play_button.text = 'Play'
        self.manager.get_screen('menu').label_settings_button.text = 'Settings'
        self.label_settings.text = 'Settings'
        self.settings_menu_button.text = 'Menu'
        self.manager.get_screen('game').label_source.text = 'Source'
        self.manager.get_screen('game').label_expert_mode.text = 'Expert Mode'
        self.manager.get_screen('game').label_eavesdropping.text = 'Eavesdropping'
        self.manager.get_screen('game').game_menu_button.text = 'Menu'
        self.manager.get_screen('game').tut_press_source.text = 'Press me!'
        self.manager.get_screen('game').tut_emmited.text = 'You have emitted a pair\nof photons!'
        self.manager.get_screen('game').tut_alice_button.text = "Press here to change\nAlice's basis"
        self.manager.get_screen('game').tut_bob_button.text = "And here to change\nBob's basis"
        self.manager.get_screen('game').tut_alice_bit.text = "These are Alice's bits"
        self.manager.get_screen('game').tut_bob_bit.text = "And these Bob's bits"
        self.manager.get_screen('game').tut_eve.text = "Press here to add an eavesdropper"
        self.manager.get_screen('game').tut_eve_left.text = 'Eve can hear from the left'
        self.manager.get_screen('game').tut_eve_right.text = 'And from the right'
        self.manager.get_screen('game').tut_slider.text = 'Slide to choose how many\npairs you send'
        self.manager.get_screen('game').tut_eve_expert.text = 'Press here for details'


class E91Simulation(Screen):

    def __init__(self, **kwargs):
        super(E91Simulation, self).__init__(**kwargs)
        Clock.schedule_interval(self.update, 1.0 / 60.0)

    source = ObjectProperty(None)

    photon_alice = ObjectProperty(None)
    photon_bob = ObjectProperty(None)

    alice_filter = ObjectProperty(None)
    bob_filter = ObjectProperty(None)
    eve_left_filter = ObjectProperty(None)
    eve_right_filter = ObjectProperty(None)

    alice_slit = ObjectProperty(None)
    bob_slit = ObjectProperty(None)
    eve_left_slit = ObjectProperty(None)
    eve_right_slit = ObjectProperty(None)

    a_bit_old = NumericProperty(0)
    b_bit_old = NumericProperty(0)
    e_left_bit_old = NumericProperty(0)
    e_right_bit_old = NumericProperty(0)
    
    a_bit = NumericProperty(0)
    b_bit = NumericProperty(0)
    e_left_bit = NumericProperty(0)
    e_right_bit = NumericProperty(0)

    switch_eve = ObjectProperty(None)
    switch_eve_left = ObjectProperty(None)
    switch_eve_right = ObjectProperty(None)
    switch_expert_mode = ObjectProperty(None)

    slider = ObjectProperty(None)

    # Counters
    counts_a1b1 = np.zeros((2,2))
    counts_a1b3 = np.zeros((2,2))
    counts_a3b1 = np.zeros((2,2))
    counts_a3b3 = np.zeros((2,2))

    # Probabilities
    prob_a1b1 = np.zeros((2,2))
    prob_a1b3 = np.zeros((2,2))
    prob_a3b1 = np.zeros((2,2))
    prob_a3b3 = np.zeros((2,2))

    # Correlation Coefficients
    E_a1b1 = NumericProperty(0.0)
    E_a1b3 = NumericProperty(0.0)
    E_a3b1 = NumericProperty(0.0)
    E_a3b3 = NumericProperty(0.0)

    S = NumericProperty(0.0)


    def source_animation(self, widget):
        # original size_hint_x = 0.04
        anim  = Animation(size_hint_x=0.045, size_hint_y=0.045*self.width/self.height, duration=0.4)
        anim += Animation(size_hint_x=0.04, size_hint_y=0.04*self.width/self.height, duration=0.4)
        anim.repeat = True
        anim.start(widget)

    def filter_animation(self, widget):
        # original size_hint_x = 0.15
        anim  = Animation(size_hint_x=0.155, size_hint_y=0.155*self.width/self.height, duration=0.4)
        anim += Animation(size_hint_x=0.15, size_hint_y=0.15*self.width/self.height, duration=0.4)
        anim.repeat = True
        anim.start(widget)

    def switch_animation(self, widget):
        # original size_hint = 0.045, 0.035
        anim  = Animation(size_hint_x=0.05, size_hint_y=0.04, duration=0.4)
        anim += Animation(size_hint_x=0.045, size_hint_y=0.035, duration=0.4)
        anim.repeat = True
        anim.start(widget)

    ''' TBD later or never
    def cursor_animation(self, widget):
        # original cursor_size: 0.025*game_layout.height, 0.025*game_layout.height
        if 0.023 * self.height < widget.cursor_width < 0.03 * self.height:
            widget.cursor_size = 0.021 * self.height, 0.021 * self.height 
        elif 0.019 * self.height < widget.cursor_width < 0.022 * self.height:
            widget.cursor_size = 0.025 * self.height, 0.025 * self.height
    '''

    def slider_animation(self, widget):
        # original size_hint = 0.1, 0.04
        anim  = Animation(size_hint_x=0.107, duration=0.4)
        anim += Animation(size_hint_x=0.1, duration=0.4)
        anim.repeat = True
        anim.start(widget)

    def stop_animation(self, widget):
        Animation.stop_all(widget)

    # tbd
    '''
    def tutorial(self):
        next_step = True
        while next_step:
            self.source_animation(self.source)
            self.tut_press_source.opacity = 1
    '''
    
    def stop_tutorial(self):
        self.slider.value = 1
        self.alice_slit.angle = 0
        self.bob_slit.angle = 0
        self.eve_left_slit.angle = 0
        self.eve_right_slit.angle = 0
        self.switch_eve.active = False
        self.switch_eve_left.active = False
        self.switch_eve_right.active = False
        self.switch_expert_mode.active = False
        Animation.stop_all(self.source)
        Animation.stop_all(self.alice_filter)
        Animation.stop_all(self.bob_filter)
        Animation.stop_all(self.switch_eve)
        Animation.stop_all(self.switch_eve_left)
        Animation.stop_all(self.switch_eve_right)
        Animation.stop_all(self.slider)
        Animation.stop_all(self.switch_expert_mode)
        self.tut_press_source.opacity = 0
        self.tut_emmited.opacity = 0
        self.tut_alice_button.opacity = 0
        self.tut_bob_button.opacity = 0
        self.tut_alice_bit.opacity = 0
        self.tut_bob_bit.opacity = 0
        self.tut_eve.opacity = 0
        self.tut_eve_left.opacity = 0
        self.tut_eve_right.opacity = 0
        self.tut_slider.opacity = 0
        self.tut_eve_expert.opacity = 0


    def emission(self):

        self.photon_alice.center = self.center
        self.photon_bob.center   = self.center

        velocity_x = (self.center_x - self.alice_filter.center_x)/60
        velocity_y = 0.5 * self.photon_alice.height

        self.photon_alice.velocity = Vector(-velocity_x, velocity_y)
        self.photon_bob.velocity   = Vector(velocity_x, velocity_y)

        basis_a   = self.alice_slit.angle_to_basis()
        basis_b   = self.bob_slit.angle_to_basis()
        basis_e_a = self.eve_left_slit.angle_to_basis()
        basis_e_b = self.eve_right_slit.angle_to_basis()

        for _ in range(0, self.slider.value):

            if self.switch_eve.active == False:
                self.a_bit_old, self.b_bit_old = measure_polarization(basis_a, basis_b)
            elif self.switch_eve_left.active and self.switch_eve_right.active:
                self.a_bit_old, self.b_bit_old, self.e_left_bit_old, self.e_right_bit_old = measure_polarization_eavesdropping(basis_a, basis_b, basis_e_a, basis_e_b)
            elif self.switch_eve_left.active:
                self.a_bit_old, self.b_bit_old, self.e_left_bit_old = measure_polarization_eavesdropping_left(basis_a, basis_b, basis_e_a)
            elif self.switch_eve_right.active:
                self.a_bit_old, self.b_bit_old, self.e_right_bit_old = measure_polarization_eavesdropping_right(basis_a, basis_b, basis_e_b)
        
            if (basis_a, basis_b) == (ZERO_BASIS, MINUS_PI_8_BASIS):
                self.counts_a1b1[self.a_bit_old, self.b_bit_old] += 1
                self.prob_a1b1 = self.counts_a1b1 / np.sum(self.counts_a1b1)
                self.E_a1b1 = round(float(self.prob_a1b1[0,0] + self.prob_a1b1[1,1] - self.prob_a1b1[1,0] - self.prob_a1b1[0,1]), 4)
            elif (basis_a, basis_b) == (ZERO_BASIS, PI_8_BASIS):
                self.counts_a1b3[self.a_bit_old, self.b_bit_old] += 1
                self.prob_a1b3 = self.counts_a1b3 / np.sum(self.counts_a1b3)
                self.E_a1b3 = round(float(self.prob_a1b3[0,0] + self.prob_a1b3[1,1] - self.prob_a1b3[1,0] - self.prob_a1b3[0,1]), 4)
            elif (basis_a, basis_b) == (PI_4_BASIS, MINUS_PI_8_BASIS):
                self.counts_a3b1[self.a_bit_old, self.b_bit_old] += 1
                self.prob_a3b1 = self.counts_a3b1 / np.sum(self.counts_a3b1)
                self.E_a3b1 = round(float(self.prob_a3b1[0,0] + self.prob_a3b1[1,1] - self.prob_a3b1[1,0] - self.prob_a3b1[0,1]), 4)
            elif (basis_a, basis_b) == (PI_4_BASIS, PI_8_BASIS):
                self.counts_a3b3[self.a_bit_old, self.b_bit_old] += 1
                self.prob_a3b3 = self.counts_a3b3 / np.sum(self.counts_a3b3)
                self.E_a3b3 = round(float(self.prob_a3b3[0,0] + self.prob_a3b3[1,1] - self.prob_a3b3[1,0] - self.prob_a3b3[0,1]), 4)

        self.S = round(self.E_a1b1 + self.E_a1b3 - self.E_a3b1 + self.E_a3b3, 4)


    def update(self, dt):
        
        self.photon_alice.move()
        self.photon_bob.move()

        # For each frame, vertical velocity is inverted 
        if self.photon_alice.velocity_x != 0:
            self.photon_alice.velocity_y *= -1 
            self.photon_bob.velocity_y   *= -1

        if OFFSET < self.eve_left_filter.center_x - self.photon_alice.x < 10 * OFFSET:
            if self.switch_eve_left.active and self.switch_eve.active:
                self.photon_alice.opacity = 0

            if self.switch_eve_right.active and self.switch_eve.active:
                self.photon_bob.opacity = 0

            self.e_left_bit = self.e_left_bit_old
            self.e_right_bit = self.e_right_bit_old
        else:
            self.photon_alice.opacity = 1
            self.photon_bob.opacity = 1

        if self.photon_alice.x < self.alice_filter.center_x - OFFSET:
            self.photon_alice.center = self.center
            self.photon_bob.center = self.center
            self.photon_alice.velocity = Vector(0, 0)
            self.photon_bob.velocity = Vector(0, 0)
            self.a_bit = self.a_bit_old
            self.b_bit = self.b_bit_old


        '''
        if self.tut_slider.opacity == 1:
            if self.big_cursor == False:
                self.slider.cursor_size = 0.025*self.height, 0.025*self.height
            else:
                self.slider.cursor_size = 0.02*self.height, 0.02*self.height
            time.sleep(0.4)
        '''

        

class Ekert91App(App): 
    
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(SettingsScreen(name='settings'))
        sm.add_widget(E91Simulation(name='game'))

        return sm

if __name__=='__main__':

    Window.maximize()
    #Window.fullscreen = 'auto'  

    Ekert91App().run() 