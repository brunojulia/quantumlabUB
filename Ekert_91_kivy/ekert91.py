import numpy as np
from QKD_functions import *

from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
from kivy.core.window import Window
from kivy.clock import Clock

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.animation import Animation
from kivy.vector import Vector
from kivy.uix.screenmanager import ScreenManager, Screen


from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, BooleanProperty

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


class SettingsScreen(Screen):
        
    def catalan(self):
        self.manager.get_screen('menu').label_play_button.text = 'Jugar'
        self.manager.get_screen('menu').label_settings_button.text = 'Ajustaments'
        self.manager.get_screen('settings').label_settings.text = 'Ajustaments'
        self.manager.get_screen('settings').settings_menu_button.text = 'Menú'
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
        self.manager.get_screen('settings').label_settings.text = 'Ajustes'
        self.manager.get_screen('settings').settings_menu_button.text = 'Menú'
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
        self.manager.get_screen('settings').label_settings.text = 'Settings'
        self.manager.get_screen('settings').settings_menu_button.text = 'Menu'
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

    photon_alice_1 = ObjectProperty(None)
    photon_alice_2 = ObjectProperty(None)
    photon_alice_3 = ObjectProperty(None)
    photon_alice_4 = ObjectProperty(None)
    photon_alice_5 = ObjectProperty(None)
    photon_alice_6 = ObjectProperty(None)
    photon_alice_7 = ObjectProperty(None)
    photon_alice_8 = ObjectProperty(None)
    photon_alice_9 = ObjectProperty(None)
    photon_alice_10 = ObjectProperty(None)

    photons_alice = ReferenceListProperty(
        photon_alice_1,
        photon_alice_2, 
        photon_alice_3,
        photon_alice_4,
        photon_alice_5,
        photon_alice_6,
        photon_alice_7, 
        photon_alice_8, 
        photon_alice_9, 
        photon_alice_10 )

    photon_bob_1 = ObjectProperty(None)
    photon_bob_2 = ObjectProperty(None)
    photon_bob_3 = ObjectProperty(None)
    photon_bob_4 = ObjectProperty(None)
    photon_bob_5 = ObjectProperty(None)
    photon_bob_6 = ObjectProperty(None)
    photon_bob_7 = ObjectProperty(None)
    photon_bob_8 = ObjectProperty(None)
    photon_bob_9 = ObjectProperty(None)
    photon_bob_10 = ObjectProperty(None)

    photons_bob = ReferenceListProperty(
        photon_bob_1,
        photon_bob_2, 
        photon_bob_3,
        photon_bob_4,
        photon_bob_5,
        photon_bob_6,
        photon_bob_7, 
        photon_bob_8, 
        photon_bob_9, 
        photon_bob_10 )

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

    slider_emission_value = NumericProperty(1)


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

        self.manager.get_screen('game').tut_press_source.opacity = 0
        self.manager.get_screen('game').tut_emmited.opacity = 0
        self.manager.get_screen('game').tut_alice_button.opacity = 0
        self.manager.get_screen('game').tut_bob_button.opacity = 0
        self.manager.get_screen('game').tut_alice_bit.opacity = 0
        self.manager.get_screen('game').tut_bob_bit.opacity = 0
        self.manager.get_screen('game').tut_eve.opacity = 0
        self.manager.get_screen('game').tut_eve_left.opacity = 0
        self.manager.get_screen('game').tut_eve_right.opacity = 0
        self.manager.get_screen('game').tut_slider.opacity = 0
        self.manager.get_screen('game').tut_eve_expert.opacity = 0

        self.counts_a1b1 = np.zeros((2,2))
        self.counts_a1b3 = np.zeros((2,2))
        self.counts_a3b1 = np.zeros((2,2))
        self.counts_a3b3 = np.zeros((2,2))
        self.prob_a1b1 = np.zeros((2,2))
        self.prob_a1b3 = np.zeros((2,2))
        self.prob_a3b1 = np.zeros((2,2))
        self.prob_a3b3 = np.zeros((2,2))
        self.E_a1b1 = 0.0
        self.E_a1b3 = 0.0
        self.E_a3b1 = 0.0
        self.E_a3b3 = 0.0
        self.S = 0.0

    def clear(self):
        self.counts_a1b1 = np.zeros((2,2))
        self.counts_a1b3 = np.zeros((2,2))
        self.counts_a3b1 = np.zeros((2,2))
        self.counts_a3b3 = np.zeros((2,2))
        self.prob_a1b1 = np.zeros((2,2))
        self.prob_a1b3 = np.zeros((2,2))
        self.prob_a3b1 = np.zeros((2,2))
        self.prob_a3b3 = np.zeros((2,2))
        self.E_a1b1 = 0.0
        self.E_a1b3 = 0.0
        self.E_a3b1 = 0.0
        self.E_a3b3 = 0.0
        self.S = 0.0


    def emission(self):

        self.slider_emission_value = self.slider.value

        velocity_x = (self.center_x - self.alice_filter.center_x)/60
        velocity_y = 0.5 * self.photon_alice_1.height

        for i in range(10):
            
            self.photons_alice[i].center_x = self.center_x + 1.1 * self.photon_alice_1.width * i
            self.photons_alice[i].center_y = self.center_y + np.random.randint(low=-0.25*self.source.height, high=0.25*self.source.height)

            self.photons_bob[i].center_x = self.center_x - 1.1 * self.photon_alice_1.width * i
            #self.photons_bob[i].center_y = self.center_y + np.random.randint(low=-0.25*self.source.height, high=0.25*self.source.height)
            self.photons_bob[i].center_y = self.photons_alice[i].center_y

            self.photons_alice[i].velocity = Vector(-velocity_x, velocity_y)
            self.photons_bob[i].velocity   = Vector(velocity_x, velocity_y)

        basis_a   = self.alice_slit.angle_to_basis()
        basis_b   = self.bob_slit.angle_to_basis()
        basis_e_a = self.eve_left_slit.angle_to_basis()
        basis_e_b = self.eve_right_slit.angle_to_basis()

        for _ in range(0, self.slider_emission_value):

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

        OFFSET = 0.05 * self.alice_filter.width

        for i in range(10):
            
            if self.slider_emission_value >= 100 * i + 1 and self.photons_alice[i].x < self.center_x:
                self.photons_alice[i].opacity = 1
            else:
                self.photons_alice[i].opacity = 0

            if self.slider_emission_value >= 100 * i + 1 and self.photons_bob[i].x > self.center_x:
                self.photons_bob[i].opacity = 1
            else:
                self.photons_bob[i].opacity = 0
            

            self.photons_alice[i].move()
            self.photons_bob[i].move()

            # For each frame, vertical velocity is inverted 
            if self.photons_alice[i].velocity_x != 0:
                self.photons_alice[i].velocity_y *= -1
                self.photons_bob[i].velocity_y *= -1
        
            if OFFSET < self.eve_left_filter.center_x - self.photons_alice[i].x < 11 * OFFSET:
                if self.switch_eve_left.active and self.switch_eve.active:
                    self.photons_alice[i].opacity = 0

                if self.switch_eve_right.active and self.switch_eve.active:
                    self.photons_bob[i].opacity = 0

                self.e_left_bit = self.e_left_bit_old
                self.e_right_bit = self.e_right_bit_old

            if self.photons_alice[i].x < self.alice_filter.center_x - OFFSET:
                self.photons_alice[i].center = self.center
                self.photons_bob[i].center = self.center
                self.photons_alice[i].velocity = Vector(0, 0)
                self.photons_bob[i].velocity = Vector(0, 0)
                self.a_bit = self.a_bit_old
                self.b_bit = self.b_bit_old


class QKeyScreen(Screen):

    def __init__(self, **kwargs):
        super(QKeyScreen, self).__init__(**kwargs)
        Clock.schedule_interval(self.update, 1.0 / 60.0)

    source = ObjectProperty(None)

    photon_alice = ObjectProperty(None)
    photon_bob   = ObjectProperty(None)

    alice_filter = ObjectProperty(None)
    bob_filter   = ObjectProperty(None)

    alice_slit = ObjectProperty(None)
    bob_slit   = ObjectProperty(None)

    a_bit_old   = NumericProperty(0)
    b_bit_old   = NumericProperty(0)
    a_basis_old = NumericProperty(0)
    b_basis_old = NumericProperty(0)
    
    a_bit   = NumericProperty(0)
    b_bit   = NumericProperty(0)
    a_basis = NumericProperty(0)
    b_basis = NumericProperty(0)

    alice_basis_1 = ObjectProperty(None)
    alice_basis_2 = ObjectProperty(None)
    alice_basis_3 = ObjectProperty(None)
    alice_basis_4 = ObjectProperty(None)
    alice_basis_5 = ObjectProperty(None)
    alice_basis_6 = ObjectProperty(None)
    alice_basis_7 = ObjectProperty(None)
    alice_basis_8 = ObjectProperty(None)
    alice_basis_9 = ObjectProperty(None)
    alice_basis_10 = ObjectProperty(None)

    alice_bases = ReferenceListProperty(
        alice_basis_1,
        alice_basis_2, 
        alice_basis_3,
        alice_basis_4,
        alice_basis_5,
        alice_basis_6,
        alice_basis_7, 
        alice_basis_8, 
        alice_basis_9, 
        alice_basis_10 )
    
    bob_basis_1 = ObjectProperty(None)
    bob_basis_2 = ObjectProperty(None)
    bob_basis_3 = ObjectProperty(None)
    bob_basis_4 = ObjectProperty(None)
    bob_basis_5 = ObjectProperty(None)
    bob_basis_6 = ObjectProperty(None)
    bob_basis_7 = ObjectProperty(None)
    bob_basis_8 = ObjectProperty(None)
    bob_basis_9 = ObjectProperty(None)
    bob_basis_10 = ObjectProperty(None)

    bob_bases = ReferenceListProperty(
        bob_basis_1,
        bob_basis_2, 
        bob_basis_3,
        bob_basis_4,
        bob_basis_5,
        bob_basis_6,
        bob_basis_7, 
        bob_basis_8, 
        bob_basis_9, 
        bob_basis_10 )
    
    alice_bit_1 = ObjectProperty(None)
    alice_bit_2 = ObjectProperty(None)
    alice_bit_3 = ObjectProperty(None)
    alice_bit_4 = ObjectProperty(None)
    alice_bit_5 = ObjectProperty(None)
    alice_bit_6 = ObjectProperty(None)
    alice_bit_7 = ObjectProperty(None)
    alice_bit_8 = ObjectProperty(None)
    alice_bit_9 = ObjectProperty(None)
    alice_bit_10 = ObjectProperty(None)

    alice_bits = ReferenceListProperty(
        alice_bit_1,
        alice_bit_2, 
        alice_bit_3,
        alice_bit_4,
        alice_bit_5,
        alice_bit_6,
        alice_bit_7, 
        alice_bit_8, 
        alice_bit_9, 
        alice_bit_10 )
    
    bob_bit_1 = ObjectProperty(None)
    bob_bit_2 = ObjectProperty(None)
    bob_bit_3 = ObjectProperty(None)
    bob_bit_4 = ObjectProperty(None)
    bob_bit_5 = ObjectProperty(None)
    bob_bit_6 = ObjectProperty(None)
    bob_bit_7 = ObjectProperty(None)
    bob_bit_8 = ObjectProperty(None)
    bob_bit_9 = ObjectProperty(None)
    bob_bit_10 = ObjectProperty(None)

    bob_bits = ReferenceListProperty(
        bob_bit_1,
        bob_bit_2, 
        bob_bit_3,
        bob_bit_4,
        bob_bit_5,
        bob_bit_6,
        bob_bit_7, 
        bob_bit_8, 
        bob_bit_9, 
        bob_bit_10 )
    
    key_bit_1 = ObjectProperty(None)
    key_bit_2 = ObjectProperty(None)
    key_bit_3 = ObjectProperty(None)
    key_bit_4 = ObjectProperty(None)
    key_bit_5 = ObjectProperty(None)
    key_bit_6 = ObjectProperty(None)
    key_bit_7 = ObjectProperty(None)
    key_bit_8 = ObjectProperty(None)
    key_bit_9 = ObjectProperty(None)
    key_bit_10 = ObjectProperty(None)

    key_bits = ReferenceListProperty(
        key_bit_1,
        key_bit_2, 
        key_bit_3,
        key_bit_4,
        key_bit_5,
        key_bit_6,
        key_bit_7, 
        key_bit_8, 
        key_bit_9, 
        key_bit_10 )


    def clear(self):

        self.key_counter = 0

        for i in range(0,10):
            self.alice_bases[i].opacity = 0
            self.bob_bases[i].opacity = 0
            self.alice_bits[i].opacity = 0
            self.bob_bits[i].opacity = 0
            self.key_bits[i].opacity = 0
            self.alice_bits[i].color = (0, 0, 0, 1)
            self.bob_bits[i].color = (0, 0, 0, 1)
            self.key_bits[i].color = (0, 0, 0, 1)

            #Clock.cancel()
        

    key_counter = 0

    def update_table_col(self, i: int, a_basis, b_basis, a_bit_old, b_bit_old):
        self.alice_bases[i].source = self.basis_to_image_black(a_basis)
        self.bob_bases[i].source = self.basis_to_image_black(b_basis)
        self.alice_bits[i].text = str(a_bit_old)
        self.bob_bits[i].text = str(b_bit_old)
        self.alice_bases[i].opacity = 1
        self.bob_bases[i].opacity = 1
        self.alice_bits[i].opacity = 1
        self.bob_bits[i].opacity = 1

        if (a_basis == b_basis):
            self.alice_bases[i].source = self.basis_to_image_blue(a_basis)
            self.bob_bases[i].source = self.basis_to_image_blue(b_basis)
            self.alice_bits[i].color = (0.203125, 0.59375, 0.855469, 1)
            self.bob_bits[i].color = (0.203125, 0.59375, 0.855469, 1)

            if (self.key_counter < 10):

                self.key_bits[self.key_counter].text = str(a_bit_old)
                self.key_bits[self.key_counter].opacity = 1
                self.key_bits[self.key_counter].color = (0.203125, 0.59375, 0.855469, 1)


                if (self.key_counter == 9):
                    for i in range(0,10):
                        self.key_bits[i].color = (160/255, 32/255, 240/255, 1)

                self.key_counter += 1 


    def basis_to_angle(self, basis):

        match basis: 
            case -1:
                return -22.5
            case 0:
                return 0
            case 1:
                return 22.5
            case 2:
                return 45
            
    def basis_to_image_black(self, basis):

        match basis: 
            case -1:
                return "Filters/filter_black_minus_225.png"
            case 0:
                return "Filters/filter_black_0.png"
            case 1:
                return "Filters/filter_black_225.png"
            case 2:
                return "Filters/filter_black_45.png"
            
    def basis_to_image_blue(self, basis):

        match basis: 
            case 0:
                return "Filters/filter_blue_0.png"
            case 1:
                return "Filters/filter_blue_225.png"
            
    steps = 35

    def emission(self, i: int):  

        OFFSET = 0.05 * self.alice_filter.width
        
        velocity_x = (self.center_x - self.alice_filter.center_x + OFFSET)/self.steps
        velocity_y = 0.5 * self.photon_alice.height
            
        self.photon_alice.center_x = self.source.center_x
        self.photon_alice.center_y = self.source.center_y # + np.random.randint(low=-0.2*self.source.height, high=0.2*self.source.height)

        self.photon_bob.center_x = self.source.center_x
        self.photon_bob.center_y = self.photon_alice.center_y

        self.photon_alice.velocity = Vector(-velocity_x, velocity_y)
        self.photon_bob.velocity   = Vector(velocity_x, velocity_y)

        self.a_basis, self.b_basis, self.a_bit_old, self.b_bit_old = random_measure_polarization()

        self.alice_slit.angle = self.basis_to_angle(self.a_basis)
        self.bob_slit.angle = self.basis_to_angle(self.b_basis)

        Clock.schedule_once(lambda dt: self.update_table_col(i, self.a_basis, self.b_basis, self.a_bit_old, self.b_bit_old), self.steps/60)


    def able_source(self):
        self.source.disabled = False

    def sequence_emission(self):

        for i in range(0,10):
            self.alice_bases[i].opacity = 0
            self.bob_bases[i].opacity = 0
            self.alice_bits[i].opacity = 0
            self.bob_bits[i].opacity = 0
            self.alice_bits[i].color = (0, 0, 0, 1)
            self.bob_bits[i].color = (0, 0, 0, 1)
            self.key_bits[i].color = (0, 0, 0, 1)

        if (self.key_counter > 9):
            self.key_counter = 0
            for i in range(0,10):
                self.key_bits[i].opacity = 0
                self.bob_bases[i].opacity = 0
                self.alice_bits[i].opacity = 0
                self.bob_bits[i].opacity = 0


        delay = self.steps/60 + 0.1

        self.source.disabled = True
        self.emission(0)
        Clock.schedule_once(lambda dt: self.emission(1), 1 * delay)
        Clock.schedule_once(lambda dt: self.emission(2), 2 * delay)
        Clock.schedule_once(lambda dt: self.emission(3), 3 * delay)
        Clock.schedule_once(lambda dt: self.emission(4), 4 * delay)
        Clock.schedule_once(lambda dt: self.emission(5), 5 * delay)
        Clock.schedule_once(lambda dt: self.emission(6), 6 * delay)
        Clock.schedule_once(lambda dt: self.emission(7), 7 * delay)
        Clock.schedule_once(lambda dt: self.emission(8), 8 * delay)
        Clock.schedule_once(lambda dt: self.emission(9), 9 * delay)
        Clock.schedule_once(lambda dt: self.able_source(), 10 * delay + 0.25)
        #for i in range(1,10):
        #    Clock.schedule_once(lambda dt: self.emission(i), i * 1.2)




    def update(self, dt):

        OFFSET = 0.05 * self.alice_filter.width

        self.photon_alice.move()
        self.photon_bob.move()

        # For each frame, vertical velocity is inverted 
        if self.photon_alice.velocity_x != 0:
            self.photon_alice.velocity_y *= -1
            self.photon_bob.velocity_y *= -1

        if self.photon_alice.x < self.alice_filter.center_x - OFFSET:
            self.photon_alice.center = self.source.center
            self.photon_bob.center = self.source.center
            self.photon_alice.velocity = Vector(0, 0)
            self.photon_bob.velocity = Vector(0, 0)
            self.a_bit = self.a_bit_old
            self.b_bit = self.b_bit_old



class Ekert91App(App): 
    
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(SettingsScreen(name='settings'))
        sm.add_widget(E91Simulation(name='game'))
        sm.add_widget(QKeyScreen(name='key'))

        return sm

if __name__=='__main__':

    Window.maximize()
    #Window.fullscreen = 'auto'

    Ekert91App().run()