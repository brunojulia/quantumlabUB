import matplotlib
import csv
import concurrent.futures
from multiprocessing import freeze_support
import functools
import itertools
import threading
from numpy import arange, sin, pi
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
import numpy as np
import time

# kivy imports
if __name__ == '__main__':  # to avoid new window with a new process
    freeze_support()  # support multiprocessing in pyinstaller

    matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
    import matplotlib.pyplot as plt

    plt.switch_backend('agg')  # very important, otherwise the threading won't work for some reason...
    # solves left click issue
    from kivy.config import Config

    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
    from kivy.animation import Animation
    from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas, FigureCanvasKivyAgg
    from matplotlib.figure import Figure
    from kivy.uix.behaviors import ButtonBehavior
    from kivy.app import App
    from kivy.clock import Clock, mainthread
    from kivy.uix.screenmanager import ScreenManager, Screen
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.floatlayout import FloatLayout
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.label import Label
    from kivy.uix.textinput import TextInput
    from kivy.uix.widget import Widget
    from kivy.uix.switch import Switch
    from kivy.uix.button import Button
    from kivy.uix.checkbox import CheckBox
    from kivy.garden.knob import Knob
    from kivy.graphics import Rectangle, Color, Line, PushMatrix, PopMatrix, Scale, Translate
    from kivy.lang import Builder
    from kivy.uix.popup import Popup
    from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas, \
        NavigationToolbar2Kivy
    from kivy.clock import Clock

    from kivy.properties import ObjectProperty, NumericProperty, StringProperty, ListProperty, BooleanProperty, \
        OptionProperty
    import math

    from entangledexp import entangledEXP  # importa les funcions que tenen a veure amb l'experiment.

if __name__ == '__main__':  # to avoid new window with a new process
    freeze_support()  # support multiprocessing in pyinstaller


    class TrueScreen(ScreenManager):
        pass


    # Defining the screenmanager otherwise TablePopup doesn't work
    sm = TrueScreen()


    class TablePopup(Screen):
        g_rectangle = ObjectProperty()
        table = ListProperty()
        table_lay = ObjectProperty()

        def __init__(self, *args, **kwargs):
            super(TablePopup, self).__init__(**kwargs)
            self.table = sm.get_screen('ES').table

        def on_enter(self, *args):
            self.table_lay.clear_widgets()
            self.table_lay.add_widget(Button(text='\u03B1 (rad)', background_color=(0, 102 / 255, 204 / 255, 1)))
            self.table_lay.add_widget(Button(text='\u03B2 (rad)', background_color=(0, 102 / 255, 204 / 255, 1)))
            self.table_lay.add_widget(
                Button(text='N_\u03B1 (detections alpha)', background_color=(0, 102 / 255, 204 / 255, 1)))
            self.table_lay.add_widget(
                Button(text='N_\u03B2 (detections beta)', background_color=(0, 102 / 255, 204 / 255, 1)))
            self.table_lay.add_widget(Button(text='N (coincidences)', background_color=(0, 102 / 255, 204 / 255, 1)))

            self.table = sm.get_screen('ES').table
            # If a measure of the S has been taken
            if len(self.table) != 0:
                for row in self.table:
                    for item in row:
                        label_i = Button(text=str(round(item, 2)), size_hint=(1, 1),
                                         background_color=(0, 102 / 255, 204 / 255, 0.8))

                        self.table_lay.add_widget(label_i)


    class EntangledScreen(Screen):
        # Quant variables:
        table = ListProperty()
        n_label = ObjectProperty()
        label_s1 = ObjectProperty()
        label_s2 = ObjectProperty()
        textin_alph = ObjectProperty()
        textin_bet = ObjectProperty()
        s_label = ObjectProperty()
        table_checkbox = ObjectProperty(Switch)
        table_popup = ObjectProperty()
        graph_checkbox = ObjectProperty(Switch)
        select_button = ObjectProperty()
        delete_button = ObjectProperty()
        plot_btn = ObjectProperty()
        clear_btn = ObjectProperty()
        b1_label = ObjectProperty()
        b2_label = ObjectProperty()
        b1_val = NumericProperty()
        b2_val = NumericProperty()
        kwinput = ObjectProperty()
        img_widg = ObjectProperty()
        angle_1 = NumericProperty(0)
        angle_2 = NumericProperty(0)
        tab_change = BooleanProperty()

        # Cla variables:

        rho1 = ObjectProperty(Switch)
        rho2 = ObjectProperty(Switch)
        n_label_hvt = ObjectProperty()
        label_s1_hvt = ObjectProperty()
        label_s2_hvt = ObjectProperty()
        textin_alph_hvt = ObjectProperty()
        textin_bet_hvt = ObjectProperty()
        table_checkbox_hvt = ObjectProperty()
        s_label_hvt = ObjectProperty()
        delete_button_hvt = ObjectProperty()
        select_button_hvt = ObjectProperty()
        b1_label_hvt = ObjectProperty()
        b2_label_hvt = ObjectProperty()
        reps = ObjectProperty()
        alpha_hvt = ObjectProperty()
        rho_select = NumericProperty()
        changed_rho = BooleanProperty()
        sum_btn = ObjectProperty(Button)
        rest_btn = ObjectProperty(Button)
        plot_btn_hvt = ObjectProperty(Button)

        # Animation props

        line_b_pos = ListProperty([[760, 680], [760, 680]])
        line_r_1_pos = ListProperty([[760, 680], [760, 680]])
        line_r_2_pos = ListProperty([[760, 680], [760, 680]])
        joint = OptionProperty('none', options=('round', 'miter', 'bevel', 'none'))
        cap = OptionProperty('none', options=('round', 'square', 'none'))
        linewidth = NumericProperty(2)
        dash_length = NumericProperty(1)
        dash_offset = NumericProperty(0)
        dashes = ListProperty([])

        # real data props

        file_name = StringProperty()

        def __init__(self, angle_count=0, tab_selector=0, angle_count_hvt=0, **kwargs):
            super(EntangledScreen, self).__init__()
            self.angle_count = angle_count
            self.angle_count_hvt = angle_count_hvt
            self.kwinput = False
            self.tab_selector = tab_selector
            self.experiment = entangledEXP()
            self.table_checkbox.bind(active=self.on_checkbox_Active)  # lliga la checkbox amb la funció
            self.table_checkbox_hvt.bind(active=self.on_checkbox_Active)
            self.graph_checkbox.bind(active=self.on_graph_checkbox_Active)  # lliga la checkbox amb la funció
            self.rho1.bind(active=self.select_rho)  # lliga el switch a la funció per seleccionar la rho.
            self.rho2.bind(active=self.select_rho)
            # Canvas of the animation

            with self.img_widg.canvas.after:
                self.img_widg.rect = Rectangle(size=self.img_widg.size, pos=self.img_widg.pos,
                                               source="img/sketch_exp_2.png")
                Color(0, 0, 1)
                self.line_b = Line(points=self.img_widg.pos, joint=self.joint, cap=self.cap,
                                   width=self.linewidth, close=False, dash_length=self.dash_length,
                                   dash_offset=self.dash_offset)
                Color(1, 0, 0)
                self.line_r_1 = Line(points=self.img_widg.pos, joint=self.joint, cap=self.cap,
                                     width=self.linewidth, close=False, dash_length=self.dash_length,
                                     dash_offset=self.dash_offset)

                self.line_r_2 = Line(points=self.img_widg.pos, joint=self.joint, cap=self.cap,
                                     width=self.linewidth, close=False, dash_length=self.dash_length,
                                     dash_offset=self.dash_offset)

        # crea el clock object que s'encarrega de fer anar l'animació
        def run_animation(self):
            Clock.unschedule(self.move_lines)
            self.line_b_pos = [[self.img_widg.pos[0] + self.img_widg.size[0] * 0.160,
                                self.img_widg.pos[1] + self.img_widg.size[1] * 0.657],
                               [self.img_widg.pos[0] + self.img_widg.size[0] * 0.160,
                                self.img_widg.pos[1] + self.img_widg.size[1] * 0.657]]

            self.line_r_1_pos = [[self.img_widg.pos[0] + self.img_widg.size[0] * 0.295,
                                  self.img_widg.pos[1] + self.img_widg.size[1] * 0.605],
                                 [self.img_widg.pos[0] + self.img_widg.size[0] * 0.295,
                                  self.img_widg.pos[1] + self.img_widg.size[1] * 0.605]]
            self.line_r_2_pos = [[self.img_widg.pos[0] + self.img_widg.size[0] * 0.295,
                                  self.img_widg.pos[1] + self.img_widg.size[1] * 0.605],
                                 [self.img_widg.pos[0] + self.img_widg.size[0] * 0.295,
                                  self.img_widg.pos[1] + self.img_widg.size[1] * 0.605]]

            self.line_b.points = self.line_b_pos
            self.line_r_1.points = self.line_r_1_pos
            self.line_r_2.points = self.line_r_2_pos

            Clock.schedule_interval(self.move_lines, 0.0001)

        # Actualitza la posició de les linies
        def move_lines(self, dt):
            initial_pos = [self.img_widg.pos[0] + self.img_widg.size[0] * 0.160,
                           self.img_widg.pos[1] + self.img_widg.size[1] * 0.657]

            end_pos_1 = [self.img_widg.pos[0] + self.img_widg.size[0] * 0.263,
                         self.img_widg.pos[1] + self.img_widg.size[1] * 0.615]

            start_pos_r_1 = [self.img_widg.pos[0] + self.img_widg.size[0] * 0.295,
                             self.img_widg.pos[1] + self.img_widg.size[1] * 0.605]

            start_pos_r_2 = [self.img_widg.pos[0] + self.img_widg.size[0] * 0.295,
                             self.img_widg.pos[1] + self.img_widg.size[1] * 0.605]

            end_pos_r_1 = [self.img_widg.pos[0] + self.img_widg.size[0] * 0.740,
                           self.img_widg.pos[1] + self.img_widg.size[1] * 0.515]

            end_pos_r_2 = [self.img_widg.pos[0] + self.img_widg.size[0] * 0.690,
                           self.img_widg.pos[1] + self.img_widg.size[1] * 0.395]

            # Blue line head movement
            if self.line_b_pos[1][0] < end_pos_1[0]:
                self.line_b_pos[1][0] += (initial_pos[0] + end_pos_1[0]) * 0.002 * 3.20

            if self.line_b_pos[1][1] > end_pos_1[1]:
                self.line_b_pos[1][1] -= (initial_pos[1] + end_pos_1[1]) * 0.002

            # Blue head reaches crystals
            # start red head movement
            if self.line_b_pos[1][0] >= end_pos_1[0] or self.line_b_pos[1][1] <= end_pos_1[1]:
                # extend red line 1 while it hasn't reached the polarizer
                if self.line_r_1_pos[1][0] < end_pos_r_1[0]:
                    self.line_r_1_pos[1][0] += (start_pos_r_1[0] + end_pos_r_1[0]) * 0.002 * 5
                if self.line_r_1_pos[1][1] > end_pos_r_1[1]:
                    self.line_r_1_pos[1][1] -= (start_pos_r_1[1] + end_pos_r_1[1]) * 0.002

                # extend red line 2 while it hasn't reached the polarizer
                if self.line_r_2_pos[1][0] < end_pos_r_2[0]:
                    self.line_r_2_pos[1][0] += (start_pos_r_2[0] + end_pos_r_2[0]) * 0.0045 * 1.9
                if self.line_r_2_pos[1][1] > end_pos_r_2[1]:
                    self.line_r_2_pos[1][1] -= (start_pos_r_2[1] + end_pos_r_2[1]) * 0.0045
                # if red line reaches pol blue tail advances
                if self.line_r_1_pos[1][0] >= end_pos_r_1[0] or self.line_r_1_pos[1][1] <= end_pos_r_1[1]:
                    # if blue tail hasn't reached the crystals
                    if self.line_b_pos[0][0] <= end_pos_1[0] or self.line_b_pos[0][1] >= end_pos_1[1]:
                        self.line_b_pos[0][0] += (initial_pos[0] + end_pos_1[0]) * 0.002 * 3.20
                        self.line_b_pos[0][1] -= (initial_pos[1] + end_pos_1[1]) * 0.002
                # if blue tail reaches crystals red tail advances
                if self.line_b_pos[0][0] >= end_pos_1[0] or self.line_b_pos[0][1] <= end_pos_1[1]:
                    self.line_r_1_pos[0][0] += (start_pos_r_1[0] + end_pos_r_1[0]) * 0.002 * 5
                    self.line_r_1_pos[0][1] -= (start_pos_r_1[1] + end_pos_r_1[1]) * 0.002

                    self.line_r_2_pos[0][0] += (start_pos_r_2[0] + end_pos_r_2[0]) * 0.0045 * 1.9
                    self.line_r_2_pos[0][1] -= (start_pos_r_2[1] + end_pos_r_2[1]) * 0.0045

                if self.line_r_1_pos[0][0] >= end_pos_r_1[0]:
                    self.line_r_1_pos[0][0] = self.line_r_1_pos[1][0]
                    self.line_r_1_pos[0][1] = self.line_r_1_pos[1][1]

                    self.line_r_2_pos[0][0] = self.line_r_2_pos[1][0]
                    self.line_r_2_pos[0][1] = self.line_r_2_pos[1][1]

                    Clock.unschedule(self.move_lines)

            # The tail of the line has arrived to the crystals

            self.line_b.points = self.line_b_pos
            self.line_r_1.points = self.line_r_1_pos
            self.line_r_2.points = self.line_r_2_pos

        # Adds a photons to throw
        # When the button is pressed for a long time it keeps adding photons

        def start_photon_adding(self, a):
            self.adding_phot = Clock.schedule_interval(lambda dt: self.add_photons(a), 0.2)

        def stop_photon_adding(self):
            Clock.unschedule(self.adding_phot)

        def add_photons(self, a):
            if self.tab_selector == 0:  # Qua
                self.experiment.n = int(self.n_label.text)
                if int(self.n_label.text) > 0:
                    self.experiment.addphotons(n=self.experiment.n + a)  # suma els fotons a llençar
                elif int(self.n_label.text) == 0:
                    if a > 0:
                        self.experiment.addphotons(n=self.experiment.n + a)
                self.n_label.text = str(int(self.experiment.n))

            elif self.tab_selector == 1:  # Cla
                self.experiment.n = int(self.n_label_hvt.text)
                if int(self.n_label_hvt.text) > 0:
                    self.experiment.addphotons(n=self.experiment.n + a)  # suma els fotons a llençar
                elif int(self.n_label_hvt.text) == 0:
                    if a > 0:
                        self.experiment.addphotons(n=self.experiment.n + a)
                self.n_label_hvt.text = str(int(self.experiment.n))

        # Selects rho 1 or rho 2 used in HVT tab
        def select_rho(self, checkboxInstance, isActive):
            if self.rho1.active and self.rho2.active == False:
                self.rho_select = 0
            elif self.rho2.active and self.rho1.active == False:
                self.rho_select = 1
            elif self.rho1.active and self.rho2.active:
                self.rho_select = 2
            else:
                self.rho_select = 0
                print("No distribution selected. Chosen default (0)")
            self.changed_rho = True

        def exp_thread(self):
            self.s_label_hvt.text = '[font=digital-7][color=000000][size=34] Computing S...[/color][/font][/size]'
            self.exp_th = threading.Thread(target=self.runexp)
            self.exp_th.daemon = True
            self.exp_th.start()

        # Runs the experiment. tab_selector determines if the user is in the Quantum (0) or the HVT (1) tab.
        def runexp(self):
            start = time.perf_counter()
            if self.tab_selector == 0:  # Qua
                alpha = int(
                    self.textin_alph.text) * math.pi / 180  # convertim a radians i assignem els parametres per poder fer l'experiment
                beta = int(self.textin_bet.text) * math.pi / 180
                self.experiment.photons = int(self.n_label.text)
            elif self.tab_selector == 1:  # HVT
                alpha = int(self.textin_alph_hvt.text) * math.pi / 180
                beta = int(self.textin_bet_hvt.text) * math.pi / 180
                self.experiment.photons = int(self.n_label_hvt.text)

            if self.tab_selector == 0:  # Qua

                self.table = self.experiment.expqua(alpha, beta)
                s = self.experiment.scalc(self.tab_selector, alpha, beta, 0)
                sigma = self.experiment.sigma(alpha, beta)
                print(s, "±", sigma)

                # Rounds the S and sigma decimals properly.

                rounder = sigma
                factor_counter = 0
                while rounder < 1:
                    rounder = rounder * 10
                    factor_counter += 1

                sr = round(s, factor_counter)
                sigmar = round(sigma, factor_counter)

                self.s_label.text = '[font=digital-7][color=000000][size=34] S=' + str(
                    sr) + '[/font]' + '±' + '[font=digital-7]' + str(sigmar) + '[/color][/font][/size]'

            elif self.tab_selector == 1:  # HVT
                if self.rho_select == 0 or self.rho_select == 1:
                    self.table = self.experiment.hvt(alpha, beta, self.rho_select)
                    if int(self.reps.text) > 1:
                        with concurrent.futures.ProcessPoolExecutor() as executor:
                            s_list = list(
                                executor.map(self.experiment.scalc,
                                             itertools.repeat(self.tab_selector, int(self.reps.text)),
                                             itertools.repeat(alpha, int(self.reps.text)),
                                             itertools.repeat(beta, int(self.reps.text)),
                                             itertools.repeat(self.rho_select, int(self.reps.text))))

                        s_arr = np.array(s_list)
                        sigma = s_arr.std()
                        s = s_arr.mean()

                        # Rounds the S and sigma decimals properly.
                        rounder = sigma
                        factor_counter = 0
                        while rounder < 1:
                            rounder = rounder * 10
                            factor_counter += 1
                        sr = round(s, factor_counter)
                        sigmar = round(sigma, factor_counter)
                    else:
                        s = self.experiment.scalc(self.tab_selector, alpha, beta, self.rho_select)
                        sr = round(s, 2)
                        sigmar = round(0, 2)

                    self.s_label_hvt.text = '[font=digital-7][color=000000][size=34] S=' + str(
                        sr) + '[/font]' + '±' + '[font=digital-7]' + str(sigmar) + '[/color][/font][/size]'

                elif self.rho_select == 2:
                    self.table = self.experiment.hvt(alpha, beta, 0)
                    if int(self.reps.text) > 1:
                        with concurrent.futures.ProcessPoolExecutor() as executor:
                            s_list_0 = list(
                                executor.map(self.experiment.scalc,
                                             itertools.repeat(self.tab_selector, int(self.reps.text)),
                                             itertools.repeat(alpha, int(self.reps.text)),
                                             itertools.repeat(beta, int(self.reps.text)),
                                             itertools.repeat(0, int(self.reps.text))))
                            s_list_1 = list(
                                executor.map(self.experiment.scalc,
                                             itertools.repeat(self.tab_selector, int(self.reps.text)),
                                             itertools.repeat(alpha, int(self.reps.text)),
                                             itertools.repeat(beta, int(self.reps.text)),
                                             itertools.repeat(1, int(self.reps.text))))

                        s_arr_0 = np.array(s_list_0)
                        sigma_0 = s_arr_0.std()
                        s_0 = s_arr_0.mean()

                        s_arr_1 = np.array(s_list_1)
                        sigma_1 = s_arr_1.std()
                        s_1 = s_arr_1.mean()
                        # Rounds the S and sigma decimals properly.
                        rounder = sigma_0
                        factor_counter = 0
                        while rounder < 1:
                            rounder = rounder * 10
                            factor_counter += 1
                        sr_0 = round(s_0, factor_counter)
                        sigmar_0 = round(sigma_0, factor_counter)

                        rounder = sigma_1
                        factor_counter = 0
                        while rounder < 1:
                            rounder = rounder * 10
                            factor_counter += 1
                        sr_1 = round(s_1, factor_counter)
                        sigmar_1 = round(sigma_1, factor_counter)
                    else:
                        s_0 = self.experiment.scalc(self.tab_selector, alpha, beta, 0)
                        sr_0 = round(s_0, 2)
                        sigmar_0 = round(0, 2)

                        s_1 = self.experiment.scalc(self.tab_selector, alpha, beta, 1)
                        sr_1 = round(s_1, 2)
                        sigmar_1 = round(0, 2)

                    self.s_label_hvt.text = '[font=digital-7][color=000000][size=30] S_1=' + str(
                        sr_0) + '[/font]' + '±' + '[font=digital-7]' + str(sigmar_0) + '\n' + 'S_2=' + str(
                        sr_1) + '[/font]' + '±' + '[font=digital-7]' + str(sigmar_1) + '[/color][/font][/size]'
            finish = time.perf_counter()
            print(f'finished in {round(finish - start, 2)} s')

        def open_table_popup(self):
            """opens popup window"""
            self.table_checkbox.active = False
            self.table_checkbox_hvt.active = False
            self.manager.current = 'TP'

        def close(self):

            if self.tab_selector == 0:
                self.table_popup.dismiss()
                self.table_checkbox.active = False  # reseteja la chkbox
            elif self.tab_selector == 1:
                self.table_popup_hvt.dismiss()
                self.table_checkbox_hvt.active = False  # reseteja la chkbox

        def on_checkbox_Active(self, checkboxInstance, isActive):
            if isActive:
                self.open_table_popup()

        def on_graph_checkbox_Active(self, checkboxInstance, isActive):
            self.kwinput = False
            if isActive:
                if self.tab_selector == 0:
                    self.select_button.disabled = False
                    self.delete_button.disabled = False
                    self.clear_btn.disabled = False
                    self.plot_btn.disabled = False

            if not isActive:
                if self.tab_selector == 0:
                    self.select_button.disabled = True
                    self.delete_button.disabled = True
                    self.clear_btn.disabled = True
                    self.plot_btn.disabled = True
                    self.b1_label.disabled = True
                    self.b2_label.disabled = True
                    self.angle_count = 0

        def select_angle(self):
            if self.angle_count < 2:
                self.angle_count += 1

            if self.angle_count == 1:
                self.delete_button.disabled = False
                if not self.kwinput:
                    self.b1_label.text = self.textin_bet.text

            if self.angle_count == 2:
                self.delete_button.disabled = False

                if not self.kwinput:
                    self.b2_label.text = self.textin_bet.text

            self.kwinput = False

        def delete_angle(self):

            if self.angle_count > 0:
                if self.angle_count == 1:
                    self.b1_label.text = ' '
                    self.b1_val = 0
                    self.delete_button.disabled = True
                    self.select_button.disabled = False
                if self.angle_count == 2:
                    self.b2_label.text = ' '
                    self.b2_val = 0
                    self.select_button.disabled = False
                self.angle_count -= 1
            self.kwinput = False

        def angle_up_hvt(self):
            if int(self.alpha_hvt.text) < 360:
                self.alpha_hvt.text = str(int(self.alpha_hvt.text) + 1)
            if int(self.alpha_hvt.text) == 360:
                self.alpha_hvt.text = str(0)

        def angle_down_hvt(self):
            if int(self.alpha_hvt.text) > 0:
                self.alpha_hvt.text = str(int(self.alpha_hvt.text) - 1)
            if int(self.alpha_hvt.text) == 0:
                self.alpha_hvt.text = str(360)

        def start_angle_up(self):
            self.adding_angle = Clock.schedule_interval(lambda dt: self.angle_up_hvt(), 0.18)

        def stop_angle_up(self):
            Clock.unschedule(self.adding_angle)

        def start_angle_down(self):
            self.subtracting_angle = Clock.schedule_interval(lambda dt: self.angle_down_hvt(), 0.18)

        def stop_angle_down(self):
            Clock.unschedule(self.subtracting_angle)

        def clear_angles(self):
            self.b1_val = 0
            self.b2_val = 0
            if self.tab_selector == 0:
                self.b1_label.text = ' '
                self.b2_label.text = ' '
                self.delete_button.disabled = True
                self.select_button.disabled = False
                self.angle_count = 0
                self.kwinput = False

        def change_pol_angles(self):
            self.changing_angle = Clock.schedule_interval(lambda dt: self.angle_update(), 0.01)

        def stop_angles(self):
            Clock.unschedule(self.changing_angle)

        def angle_update(self):
            if self.tab_selector == 0:
                self.angle_1 = -int(self.textin_alph.text)
                self.angle_2 = -int(self.textin_bet.text)
            elif self.tab_selector == 1:
                self.angle_1 = -int(self.textin_alph_hvt.text)
                self.angle_2 = -int(self.textin_bet_hvt.text)

        def select_file(self, filename):
            try:
                self.file_name = filename[0]
                sm.current = 'DS'
                print(self.file_name)
            except:
                pass

        pass


    # AngleKnob is a knob with properties from the button such as on_release.

    class AngleKnob(ButtonBehavior, Knob):
        pass


    ############################################ Graph Layout ###############################################################

    class GraphScreen(Screen):

        mainlay = ObjectProperty()
        canv = ObjectProperty()
        alpha = NumericProperty()
        S_axis = ListProperty([])
        S_axis_1 = ListProperty([])
        S_axis_2 = ListProperty([])
        bottom_lay = ObjectProperty()

        def __init__(self, *args, **kwargs):
            super(GraphScreen, self).__init__(*args, **kwargs)
            self.exitbtn = Button(size_hint=(1, 1), text='Go Back')
            self.exitbtn.bind(on_release=self.go_back)
            self.qua_ch = CheckBox(size_hint=(0.5, 0.5), pos_hint={'x': 0.2, 'y': 0.2})
            self.qua_ch.bind(active=self.add_qua_plot)
            self.qua_lab = Label(text='Show quantum prediction', size_hint=(0.5, 0.6), pos_hint={'x': 0, 'y': 0.2})
            self.bottom_lay.add_widget(self.exitbtn, index=0)

        def graph_thread(self):
            self.gr_th = threading.Thread(target=self.get_graph)
            self.gr_th.daemon = True
            self.gr_th.start()

        def get_graph(self):

            if self.manager.get_screen('ES').tab_selector == 0:
                # avoids unnecessary mess disabling plot button
                self.manager.get_screen('ES').plot_btn.disabled = True

                # if angles changed or tab changed (HVT <--> Qua)
                if int(self.manager.get_screen('ES').b1_label.text) != self.manager.get_screen('ES').b1_val or \
                        (int(self.manager.get_screen('ES').b2_label.text) != self.manager.get_screen('ES').b2_val) or \
                        self.manager.get_screen('ES').tab_change:
                    fig = plt.figure()
                    ax = Axes3D(fig)
                    self.manager.get_screen('ES').b1_val = int(self.manager.get_screen('ES').b1_label.text)
                    self.manager.get_screen('ES').b2_val = int(self.manager.get_screen('ES').b2_label.text)

                    self.manager.get_screen('ES').angle_count = 2
                    self.manager.get_screen('ES').experiment.b1 = self.manager.get_screen('ES').b1_val * math.pi / 180
                    self.manager.get_screen('ES').experiment.b2 = self.manager.get_screen('ES').b2_val * math.pi / 180

                    (alphalist, betalist) = self.manager.get_screen('ES').experiment.sweepS()
                    scalcvec = np.vectorize(self.manager.get_screen('ES').experiment.scalc)

                    X, Y = np.meshgrid(alphalist, betalist, sparse=True)
                    Z = scalcvec(self.manager.get_screen('ES').tab_selector, X, Y, 0)

                    mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
                    mappable.set_array(Z)

                    ax.plot_surface(X, Y, Z, cmap=mappable.cmap, linewidth=0.01)

                    ax.set_xlabel('Alpha (rad)')
                    ax.set_ylabel('Beta (rad)')
                    ax.set_zlabel('S')
                    mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
                    mappable.set_array(Z)
                    cbar = fig.colorbar(mappable, shrink=0.5)
                    cbar.set_label('S', rotation=0)

                    self.canv = FigureCanvas(fig)
                    self.manager.get_screen(
                        'ES').tab_change = False  # tab_change tells if the tab has changed to replot or not
                self.add_plot()
            if self.manager.get_screen('ES').tab_selector == 1:  # HVT

                self.bottom_lay.add_widget(self.qua_ch)
                self.bottom_lay.add_widget(self.qua_lab)
                # if the angle, the rho or the tab have changed plot another figure
                if int(
                        self.manager.get_screen('ES').alpha_hvt.text) * np.pi / 180 != self.alpha or \
                        self.manager.get_screen('ES').changed_rho or self.manager.get_screen('ES').tab_change:
                    figure, ax = plt.subplots()

                    self.alpha = int(self.manager.get_screen('ES').alpha_hvt.text) * np.pi / 180
                    beta_axis = np.linspace(0, 2 * np.pi, 70)

                    # this is so the user knows the graph is being plotted

                    self.manager.get_screen('ES').alpha_hvt.text = 'plotting, stand by...'
                    # disabling the plotting buttons so the user doesn't mess around
                    self.manager.get_screen('ES').sum_btn.disabled = True
                    self.manager.get_screen('ES').rest_btn.disabled = True
                    self.manager.get_screen('ES').plot_btn_hvt.disabled = True
                    # multiprocessing to speed up the calc of the graph
                    if self.manager.get_screen('ES').rho_select == 0 or self.manager.get_screen('ES').rho_select == 1:
                        with concurrent.futures.ProcessPoolExecutor() as executor:
                            self.S_axis = list(executor.map(self.manager.get_screen('ES').experiment.scalc,
                                                            itertools.repeat(1, len(beta_axis)),
                                                            itertools.repeat(self.alpha, len(beta_axis)), beta_axis,
                                                            itertools.repeat(self.manager.get_screen('ES').rho_select,
                                                                             len(beta_axis))))

                            ax.plot(beta_axis, self.S_axis, 'bo',
                                    label="\u03C1" + "" + str(self.manager.get_screen('ES').rho_select + 1))
                            ax.plot(beta_axis, list(itertools.repeat(2, len(beta_axis))), '-g', label='Classical limit')
                            ax.plot(beta_axis, list(itertools.repeat(-2, len(beta_axis))), '-g')

                    elif self.manager.get_screen('ES').rho_select == 2:
                        with concurrent.futures.ProcessPoolExecutor() as executor:
                            self.S_axis_1 = list(executor.map(self.manager.get_screen('ES').experiment.scalc,
                                                              itertools.repeat(1, len(beta_axis)),
                                                              itertools.repeat(self.alpha, len(beta_axis)), beta_axis,
                                                              itertools.repeat(0, len(beta_axis))))
                            self.S_axis_2 = list(executor.map(self.manager.get_screen('ES').experiment.scalc,
                                                              itertools.repeat(1, len(beta_axis)),
                                                              itertools.repeat(self.alpha, len(beta_axis)), beta_axis,
                                                              itertools.repeat(1, len(beta_axis))))

                            ax.plot(beta_axis, self.S_axis_1, "bo", label='\u03C1' + '\u2081')
                            ax.plot(beta_axis, self.S_axis_2, "ro", label='\u03C1' + '\u2082')
                            ax.plot(beta_axis, list(itertools.repeat(2, len(beta_axis))), '-g', label='Classical limit')
                            ax.plot(beta_axis, list(itertools.repeat(-2, len(beta_axis))), '-g')

                    ax.legend(loc="lower left")
                    ax.set_ylabel("S")
                    ax.set_xlabel("\u03B2 (rad)")
                    self.canv = FigureCanvasKivyAgg(figure)
                    self.manager.get_screen(
                        'ES').changed_rho = False  # changed_rho tells if the rho used has changed
                    self.manager.get_screen(
                        'ES').tab_change = False  # tab_change tells if the tab has changed to replot
                self.add_plot()

        def add_plot(self):
            self.mainlay.add_widget(self.canv, index=1)
            self.manager.current = 'GS'
            self.manager.get_screen('ES').alpha_hvt.text = str(int(self.alpha * 180 / np.pi))

            # Enabling the buttons again
            self.manager.get_screen('ES').plot_btn.disabled = False
            self.manager.get_screen('ES').sum_btn.disabled = False
            self.manager.get_screen('ES').rest_btn.disabled = False
            self.manager.get_screen('ES').plot_btn_hvt.disabled = False

        def go_back(self, instance):
            self.manager.get_screen('ES').ids.info_label.text = 'Select input angles'
            self.qua_ch.active = False
            self.manager.current = 'ES'
            self.mainlay.remove_widget(self.canv)
            self.bottom_lay.remove_widget(self.qua_ch)
            self.bottom_lay.remove_widget(self.qua_lab)

        def add_qua_plot(self, checkboxInstance, isActive):
            figure, ax = plt.subplots()
            beta_axis = np.linspace(0, 2 * np.pi, 70)
            self.mainlay.remove_widget(self.canv)
            if isActive:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    s_ax_qua = list(executor.map(self.manager.get_screen('ES').experiment.scalc,
                                                 itertools.repeat(0, len(beta_axis)),
                                                 itertools.repeat(self.alpha, len(beta_axis)), beta_axis,
                                                 itertools.repeat(self.manager.get_screen('ES').rho_select,
                                                                  len(beta_axis))))
                    if self.manager.get_screen('ES').rho_select == 0 or self.manager.get_screen('ES').rho_select == 1:
                        ax.plot(beta_axis, self.S_axis, 'bo',
                                label="\u03C1" + "" + str(self.manager.get_screen('ES').rho_select + 1))
                        ax.plot(beta_axis, list(itertools.repeat(2, len(beta_axis))), '-g', label='Classical limit')
                        ax.plot(beta_axis, list(itertools.repeat(-2, len(beta_axis))), '-g')
                        ax.plot(beta_axis, s_ax_qua, 'yo', label='Quantum prediction')

                    elif self.manager.get_screen('ES').rho_select == 2:
                        ax.plot(beta_axis, self.S_axis_1, "bo", label='\u03C1' + '\u2081')
                        ax.plot(beta_axis, self.S_axis_2, "ro", label='\u03C1' + '\u2082')
                        ax.plot(beta_axis, list(itertools.repeat(2, len(beta_axis))), '-g', label='Classical limit')
                        ax.plot(beta_axis, list(itertools.repeat(-2, len(beta_axis))), '-g')
                        ax.plot(beta_axis, s_ax_qua, 'yo', label='Quantum prediction')
                    ax.legend(loc="lower left")
                    ax.set_ylabel("S")
                    ax.set_xlabel("\u03B2 (rad)")

                    self.canv = FigureCanvasKivyAgg(figure)
                    self.manager.get_screen('ES').changed_rho = False  # changed_rho tells if the rho used has changed
                    self.manager.get_screen(
                        'ES').tab_change = False  # tab_change tells if the tab has changed to replot

                    self.add_plot()

            if not isActive:
                if self.manager.get_screen('ES').rho_select == 0 or self.manager.get_screen('ES').rho_select == 1:
                    ax.plot(beta_axis, self.S_axis, 'bo',
                            label="\u03C1" + "" + str(self.manager.get_screen('ES').rho_select + 1))
                    ax.plot(beta_axis, list(itertools.repeat(2, len(beta_axis))), '-g', label='Classical limit')
                    ax.plot(beta_axis, list(itertools.repeat(-2, len(beta_axis))), '-g')

                elif self.manager.get_screen('ES').rho_select == 2:
                    ax.plot(beta_axis, self.S_axis_1, "bo", label='\u03C1' + '\u2081')
                    ax.plot(beta_axis, self.S_axis_2, "ro", label='\u03C1' + '\u2082')
                    ax.plot(beta_axis, list(itertools.repeat(2, len(beta_axis))), '-g', label='Classical limit')
                    ax.plot(beta_axis, list(itertools.repeat(-2, len(beta_axis))), '-g')

                self.canv = FigureCanvasKivyAgg(figure)
                self.add_plot()

        pass


    class DataScreen(Screen):
        table_lay_data = ObjectProperty(GridLayout)
        table1 = ListProperty([])
        s_label_data = ObjectProperty()
        alpha = NumericProperty()
        beta = NumericProperty()

        def __init__(self, *args, **kwargs):
            super(DataScreen, self).__init__(**kwargs)

        def on_enter(self, *args):

            self.table_lay_data.clear_widgets()
            self.table_lay_data.add_widget(Button(text='\u03B1 (rad)', background_color=(0, 102 / 255, 204 / 255, 1)))
            self.table_lay_data.add_widget(Button(text='\u03B2 (rad)', background_color=(0, 102 / 255, 204 / 255, 1)))
            self.table_lay_data.add_widget(Button(text='N_a', background_color=(0, 102 / 255, 204 / 255, 1)))
            self.table_lay_data.add_widget(Button(text='N_b', background_color=(0, 102 / 255, 204 / 255, 1)))
            self.table_lay_data.add_widget(Button(text='N', background_color=(0, 102 / 255, 204 / 255, 1)))

            file_name = sm.get_screen('ES').file_name
            with open(file_name, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                # If a measure of the S has been taken
                self.table1 = []
                for row in reader:
                    table_row = []
                    for item in row:
                        label_i = Button(text=str(round(float(item), 2)), size_hint=(1, 1),
                                         background_color=(0, 102 / 255, 204 / 255, 0.8))
                        self.table_lay_data.add_widget(label_i)

                        #                   generating the table with numbers
                        table_row.append(float(item))
                    self.table1.append(table_row)
                self.alpha = self.table1[0][0]
                self.beta = self.table1[0][1]
                csvfile.close()

        def run_exp_data(self):

            s = sm.get_screen('ES').experiment.s_calc_data(self.table1)
            sigma = sm.get_screen('ES').experiment.sigma_data(self.table1)

            # Rounds the S and sigma decimals properly.

            rounder = sigma
            factor_counter = 0
            while rounder < 1:
                rounder = rounder * 10
                factor_counter += 1

            sr = round(s, factor_counter)
            sigmar = round(sigma, factor_counter)

            self.s_label_data.text = '[font=digital-7][color=000000][size=34] S=' + str(
                sr) + '[/font]' + '±' + '[font=digital-7]' + str(sigmar) + '[/color][/font][/size]'

        #   predicts HVT's S using the angles from the table

        def run_cla_pred(self):
            s_arr = np.array([])
            for rep in range(0, 10):
                s_i = sm.get_screen('ES').experiment.scalc(1, self.alpha, self.beta, 0)
                s_arr = np.append(s_arr, s_i)
            sigma = s_arr.std()
            s = s_arr.mean()

            # Rounds the S and sigma decimals properly.
            rounder = sigma
            factor_counter = 0
            while rounder < 1:
                rounder = rounder * 10
                factor_counter += 1

            sr = round(s, factor_counter)
            sigmar = round(sigma, factor_counter)

            self.s_label_data.text = '[font=digital-7][color=000000][size=34] S=' + str(
                sr) + '[/font]' + '±' + '[font=digital-7]' + str(sigmar) + '[/color][/font][/size]'

            return sr, " ± ", sigmar

        pass


    class InfoScreen(Screen):
        top_ch = ObjectProperty(CheckBox)
        trb_ch = ObjectProperty(CheckBox)
        right_ch = ObjectProperty(CheckBox)
        tlf_ch = ObjectProperty(CheckBox)
        front_ch = ObjectProperty(CheckBox)
        trf_ch = ObjectProperty(CheckBox)
        im_view = ObjectProperty()
        comp_pos_im = ObjectProperty()
        comp_im = ObjectProperty()
        defs_dict = ObjectProperty()
        specs_lab = ObjectProperty(Label)

        def __init__(self, *args, **kwargs):
            super(InfoScreen, self).__init__()

        # loads the components' descriptions from a file when the user enters the screen and creates a dictionary with
        # the item's number and the text. Then, on_comp_select changes the label's text according to the component selected.

        def on_enter(self, *args):
            line_num = 0
            par_num = 1
            file_name = 'comp_descriptions.txt'
            with open(file_name, 'r') as comp_file:
                descriptions = comp_file.readlines()

                for line in descriptions:
                    if line == '\n':
                        descriptions[line_num] = str(par_num)
                        par_num += 1
                    line_num += 1
                lab_defs = []
                self.defs_dict = {}
                for i in descriptions:
                    if i.isnumeric():
                        self.defs_dict[i] = lab_defs
                        lab_defs = []
                    else:
                        lab_defs.append(i)
                comp_file.close()
                description = ''
            for i in self.defs_dict[str(1)]:
                description += (' ' + i)
            self.specs_lab.text = description

        # selects model view
        def on_view_select(self, instance, value, in_view_val):
            view_dict = {1: 'img/top.png', 2: 'img/top-right-back.png', 3: 'img/right.png', 4: 'img/top-left-front.png',
                         5: 'img/front.png', 6: 'img/top-right-front.png'}
            self.im_view.source = view_dict[in_view_val]

        def animate_frame(self, widget, comp_val, *args):
            anim_cry = Animation(comp_pos_y=self.comp_pos_im.size[1] * 0.2, comp_y_size=0)
            anim_pol = Animation(comp_pos_y=self.comp_pos_im.size[1] * 0.75, comp_y_size=self.comp_pos_im.size[1] * 0.2)
            anim_laser = Animation(comp_pos_y=self.comp_pos_im.size[1] * 0, comp_y_size=0)
            anim_detect = Animation(comp_pos_y=self.comp_pos_im.size[1] * 0.77,
                                    comp_y_size=self.comp_pos_im.size[1] * 0.05)

            if comp_val == 1:
                anim_laser.start(widget)
            if comp_val == 2:
                anim_cry.start(widget)
            if comp_val == 3:
                anim_pol.start(widget)
            if comp_val == 4:
                anim_detect.start(widget)

        def on_comp_select(self, instance, value, in_comp_val):
            comp_dict = {1: 'img/laser.png', 2: 'img/crystals.png', 3: 'img/quartz.png', 4: 'img/pol_real.png',
                         5: 'img/photon_counter.png'}
            self.comp_im.source = comp_dict[in_comp_val]

            description = ''
            # print(type(self.defs_dict[in_comp_val]))
            for i in self.defs_dict[str(in_comp_val)]:
                description += (' ' + i)
            self.specs_lab.text = description

        pass


    kv = Builder.load_file("entangled.kv")


    class MainApp(App):

        def build(self):
            sm.add_widget(GraphScreen(name='GS'))
            sm.add_widget(EntangledScreen(name='ES'))
            sm.add_widget(TablePopup(name='TP'))
            sm.add_widget(DataScreen(name='DS'))
            sm.add_widget(InfoScreen(name='IS'))
            sm.current = 'ES'
            return sm

if __name__ == "__main__":
    app = MainApp()
    app.run()
