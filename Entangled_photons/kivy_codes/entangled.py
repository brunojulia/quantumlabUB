import matplotlib
import csv
import concurrent.futures
from multiprocessing import Process, freeze_support
import functools
import itertools
import threading
from numpy import arange, sin, pi
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg') # very important, otherwise the threading won't work for some reason...

# kivy imports
if __name__ == '__main__':  # to avoid new window with a new process
    freeze_support()  # support multiprocessing in pyinstaller

    # solves left click issue
    from kivy.config import Config

    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

    matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
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
            super(TablePopup, self).__init__(*args, **kwargs)
            self.table = sm.get_screen('ES').table

        def on_enter(self, *args):
            self.table_lay.clear_widgets()
            self.table_lay.add_widget(Button(text='\u03B1 (rad)', background_color=(0, 102 / 255, 204 / 255, 1)))
            self.table_lay.add_widget(Button(text='\u03B2 (rad)', background_color=(0, 102 / 255, 204 / 255, 1)))
            self.table_lay.add_widget(Button(text='N_a', background_color=(0, 102 / 255, 204 / 255, 1)))
            self.table_lay.add_widget(Button(text='N_b', background_color=(0, 102 / 255, 204 / 255, 1)))
            self.table_lay.add_widget(Button(text='N', background_color=(0, 102 / 255, 204 / 255, 1)))

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
        clear_btn_hvt = ObjectProperty()
        plot_btn_hvt = ObjectProperty()
        b1_label_hvt = ObjectProperty()
        b2_label_hvt = ObjectProperty()
        reps = ObjectProperty()
        alpha_hvt = ObjectProperty()
        rho_select = NumericProperty()
        changed_rho = BooleanProperty()

        # Animation props

        line_pos = ListProperty([[760, 680], [760, 680]])
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
                self.line = Line(points=self.img_widg.pos, joint=self.joint, cap=self.cap,
                                 width=self.linewidth, close=False, dash_length=self.dash_length,
                                 dash_offset=self.dash_offset)

        # crea el clock object que s'encarrega de fer anar l'animació
        def run_animation(self):
            Clock.unschedule(self.move_lines)
            self.line_pos = [[self.img_widg.pos[0] + self.img_widg.size[0] * 0.115,
                              self.img_widg.pos[1] + self.img_widg.size[1] * 0.752],
                             [self.img_widg.pos[0] + self.img_widg.size[0] * 0.115,
                              self.img_widg.pos[1] + self.img_widg.size[1] * 0.752]]

            self.line.points = self.line_pos
            Clock.schedule_interval(self.move_lines, 1 / 60)

        # Actualitza la posició de les linies
        def move_lines(self, dt):
            initial_pos = [self.img_widg.pos[0] + self.img_widg.size[0] * 0.115,
                           self.img_widg.pos[1] + self.img_widg.size[1] * 0.752]
            end_pos_1 = [self.img_widg.pos[0] + self.img_widg.size[0] * 0.350,
                         self.img_widg.pos[1] + self.img_widg.size[1] * 0.640]
            if self.line_pos[1][0] < end_pos_1[0]:
                self.line_pos[1][0] += (initial_pos[0] + end_pos_1[0]) * 0.003 * 2.9
            if self.line_pos[1][1] > end_pos_1[1]:
                self.line_pos[1][1] -= (initial_pos[1] + end_pos_1[1]) * 0.003


            else:
                if self.line_pos[0][0] < end_pos_1[0]:
                    self.line_pos[0][0] += (initial_pos[0] + end_pos_1[0]) * 0.003 * 2.9
                if self.line_pos[0][1] > end_pos_1[1]:
                    self.line_pos[0][1] -= (initial_pos[1] + end_pos_1[1]) * 0.003
                else:
                    Clock.unschedule(self.move_lines)
            # The tail of the line has arrived to the crystals

            self.line.points = self.line_pos

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
            if self.rho1.active:
                self.rho_select = 0
            elif self.rho2.active:
                self.rho_select = 1
            else:
                self.rho_select = 0
                print("No distribution selected. Chosen default (0)")
            self.changed_rho = True

        # Runs the experiment. tab_selector determines if the user is in the Quantum (0) or the HVT (1) tab.
        def runexp(self):
            if self.tab_selector == 0:
                alpha = int(
                    self.textin_alph.text) * math.pi / 180  # convertim a radians i assignem els parametres per poder fer l'experiment
                beta = int(self.textin_bet.text) * math.pi / 180
            elif self.tab_selector == 1:
                alpha = int(self.textin_alph_hvt.text) * math.pi / 180
                beta = int(self.textin_bet_hvt.text) * math.pi / 180

            self.experiment.photons = int(self.n_label.text)

            if self.tab_selector == 0:
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

            elif self.tab_selector == 1:

                self.table = self.experiment.hvt(alpha, beta, self.rho_select)
                s_arr = np.array([])
                for rep in range(0, int(self.reps.text)):
                    s_i = self.experiment.scalc(self.tab_selector, alpha, beta, self.rho_select)
                    s_arr = np.append(s_arr, s_i)
                sigma = s_arr.std()
                s = s_arr.mean()

                # Rounds the S and sigma decimals properly.
                rounder = sigma
                factor_counter = 0
                if int(self.reps.text) > 1:
                    while rounder < 1:
                        rounder = rounder * 10
                        factor_counter += 1

                    sr = round(s, factor_counter)
                    sigmar = round(sigma, factor_counter)
                else:
                    sr = round(s, 2)
                    sigmar = round(sigma, 2)

                self.s_label_hvt.text = '[font=digital-7][color=000000][size=34] S=' + str(
                    sr) + '[/font]' + '±' + '[font=digital-7]' + str(sigmar) + '[/color][/font][/size]'

            return (sr, " ± ", sigmar)

        def open_table_popup(self):
            '''opens popup window'''
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


    ############################################ Graph Layout ################################################################

    class GraphScreen(Screen):
        mainlay = ObjectProperty()
        canv = ObjectProperty()
        alpha = NumericProperty()

        def __init__(self, *args, **kwargs):
            super(GraphScreen, self).__init__(*args, **kwargs)
            self.exitbtn = Button(size_hint=(1, 0.05), text='Go Back')
            self.exitbtn.bind(on_release=self.go_back)
            self.mainlay.add_widget(self.exitbtn, index=0)

        def graph_thread(self):
            self.gr_th = threading.Thread(target=self.get_graph)
            self.gr_th.daemon = True
            self.gr_th.start()

        def get_graph(self):

            if self.manager.get_screen('ES').tab_selector == 0:
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
                        'ES').tab_change = False  # tab_change tells if the tab has changed to replot

            if self.manager.get_screen('ES').tab_selector == 1:  # HVT

                if int(self.manager.get_screen(
                        'ES').alpha_hvt.text) * np.pi / 180 != self.alpha or self.manager.get_screen(
                        'ES').changed_rho \
                        or self.manager.get_screen('ES').tab_change:
                    figure, ax = plt.subplots()

                    self.alpha = int(self.manager.get_screen('ES').alpha_hvt.text) * np.pi / 180
                    beta_axis = np.linspace(0, 2 * np.pi, 70)
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        S_axis = list(executor.map(self.manager.get_screen('ES').experiment.scalc,
                                                   itertools.repeat(1, len(beta_axis)),
                                                   itertools.repeat(self.alpha, len(beta_axis)), beta_axis,
                                                   itertools.repeat(self.manager.get_screen('ES').rho_select,
                                                                    len(beta_axis))))

                    # for beta in beta_axis:
                    #     S_axis.append(
                    #         self.manager.get_screen('ES').experiment.scalc(1,
                    #                                                        self.alpha, beta,
                    #                                                        self.manager.get_screen('ES').rho_select))

                    ax.plot(beta_axis, S_axis)
                    ax.set_ylabel("S")
                    ax.set_xlabel("\u03B2 (rad)")

                    self.canv = FigureCanvasKivyAgg(figure)
                    self.manager.get_screen('ES').changed_rho = False  # changed_rho tells if the rho used has changed
                    self.manager.get_screen(
                        'ES').tab_change = False  # tab_change tells if the tab has changed to replot
            self.add_plot()

        def add_plot(self):
            self.mainlay.add_widget(self.canv, index=1)
            self.manager.current = 'GS'

        def go_back(self, instance):
            self.mainlay.remove_widget(self.canv)
            self.manager.current = 'ES'

        pass


    class DataScreen(Screen):
        table_lay_data = ObjectProperty(GridLayout)
        table1 = ListProperty([])
        s_label_data = ObjectProperty()
        alpha = NumericProperty()
        beta = NumericProperty()

        def __init__(self, *args, **kwargs):
            super(DataScreen, self).__init__(*args, **kwargs)

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

        pass

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

            return (sr, " ± ", sigmar)


    kv = Builder.load_file("entangled.kv")


    class MainApp(App):

        def build(self):
            sm.add_widget(GraphScreen(name='GS'))
            sm.add_widget(EntangledScreen(name='ES'))
            sm.add_widget(TablePopup(name='TP'))
            sm.add_widget(DataScreen(name='DS'))
            sm.current = 'ES'
            return sm

if __name__ == "__main__":
    app = MainApp()
    app.run()
