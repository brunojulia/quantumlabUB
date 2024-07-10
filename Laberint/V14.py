import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.image import Image as Img
from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics import Color
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.core.text import LabelBase
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.textinput import TextInput

import numpy as np
import math
import random
from PIL import Image
import os
import shutil
from kivy.clock import Clock





class LaberintApp(App):
    def build(self):
        return Inici()





class Inici(BoxLayout):
    """Primera pantalla del joc: aquí es poden llegir les instruccions o començar a jugar"""


    def __init__(self, **kwargs):
        super(Inici, self).__init__(**kwargs)

        self.orientation = "vertical"

        intruccions = Button(text='Instruccions', size_hint=(0.5, 0.25), pos_hint={'center_x': 0.5})
        intruccions.bind(on_press=self.mostrar_instruccions)
        self.add_widget(intruccions)

        tutorial = Button(text='Tutorial', size_hint=(0.5, 0.25), pos_hint={'center_x': 0.5})
        tutorial.bind(on_press=self.tutorial)
        self.add_widget(tutorial)

        jocs = Button(text='Jocs', size_hint=(0.5, 0.25), pos_hint={'center_x': 0.5})
        jocs.bind(on_press=self.jocs)
        self.add_widget(jocs)

    def jocs(self, instance):
        App.get_running_app().root.clear_widgets()
        App.get_running_app().root.add_widget(Jocs())

    def tutorial(self, instance):
        App.get_running_app().root.clear_widgets()
        App.get_running_app().root.add_widget(Tutorial())

    def mostrar_instruccions(self, instance):
        instruccions = Label(text='L\'objectiu del joc és sortir del laberint.\n'
                 'Utilitza les fletxes per a moure la partícula.\n'
                 'Però vigila! La partícula és quàntica i per tant la seva funció d\'ona patirà deformacions a mesura que passi el temps.\n'
                 'El que veuràs a la pantalla és el mòdul al quadrat de la funció d\'ona, que et donarà la probabilitat de trobar la partícula en aquella posició.\n'
                 'Prem "Mesura" per a mesurar la posició exacte de la partícula.\n'
                 )
        instruccions.bind(size=instruccions.setter('text_size'))
        scrollview = ScrollView()
        scrollview.add_widget(instruccions)
        popup = Popup(title='Instruccions',
              content=scrollview,
              size_hint=(0.5, 0.5))
        popup.open()





class Tutorial(FloatLayout):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        imagen = Image.open("tutorial.jpg")
        self.add_widget(Finestra(imagen, 30, 14, 27, 14, 1))

        





class Jocs(GridLayout):

    def __init__(self, **kwargs):
        super(Jocs, self).__init__(**kwargs)

        self.cols = 2

        horta = FloatLayout()
        horta_img = Img(source='Horta_real.jpg', size_hint=(1, 0.75), pos_hint={'center_x': 0.5, 'center_y': 0.5}, keep_ratio=True, allow_stretch=True)
        horta.add_widget(horta_img)
        horta_but = Button(text='Laberint d\'Horta, Barcelona 1808', size_hint=(None,None), size=(300,30), pos_hint={'center_x': 0.5, 'center_y': 0.5}, background_color=(0,0.5,0))
        horta_but.bind(on_press=self.joc_horta)
        horta.add_widget(horta_but)
        self.add_widget(horta)

        wakefield = FloatLayout()
        wakefield_img = Img(source='wakefield.jpg', size_hint=(1, 0.75), pos_hint={'center_x': 0.5, 'center_y': 0.5}, keep_ratio=True, allow_stretch=True)
        wakefield.add_widget(wakefield_img)
        wakefield_but = Button(text='Laberint de Wakefield, Yorkshire 2013', size_hint=(None,None), size=(300,30), pos_hint={'center_x': 0.5, 'center_y': 0.5}, background_color=(0,0.5,0))
        wakefield_but.bind(on_press=self.joc_wakefield)
        wakefield.add_widget(wakefield_but)
        self.add_widget(wakefield)

        scone = FloatLayout()
        scone_img = Img(source='Scone_real.jpg', size_hint=(1, 0.75), pos_hint={'center_x': 0.5, 'center_y': 0.5}, keep_ratio=True, allow_stretch=True)
        scone.add_widget(scone_img)
        scone_but = Button(text='Laberint Murray Star, Escosia 1991', size_hint=(None,None), size=(300,30), pos_hint={'center_x': 0.5, 'center_y': 0.5}, background_color=(0,0.5,0))
        scone_but.bind(on_press=self.joc_scone)
        scone.add_widget(scone_but)
        self.add_widget(scone)

        lab4 = Button(text='Crea el teu propi laberint', size_hint=(1, 0.75), pos_hint={'center_x': 0.5, 'center_y': 0.5}, background_color=(0,0.5,0))
        lab4.bind(on_press=self.crear_lab)
        self.add_widget(lab4)

    def joc_horta(self, instance):
        imagen = Image.open("horta.png")
        self.finestra_jugar(instance, imagen, 80, 40, 40, 78, 78)
    
    def joc_wakefield(self, instance):
        imagen = Image.open("wakefield.jpg")
        self.finestra_jugar(instance, imagen, 60, 30, 30, 28, 55)

    def joc_scone(self, instance):
        imagen = Image.open("scone.jpg")
        self.finestra_jugar(instance, imagen, 80, 34, 17, 29, 15)

    def crear_lab(self, instance):
        App.get_running_app().root.clear_widgets()
        App.get_running_app().root.add_widget(CrearLab())
    
    def finestra_jugar(self, instance, imagen, nivell, x_ini, y_ini, x_f, y_f):
        App.get_running_app().root.clear_widgets()
        App.get_running_app().root.add_widget(Finestra(imagen, nivell, x_ini, y_ini, x_f, y_f))





class CrearLab(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.orientation = "vertical"
        self.fitxer = None
        self.mida = None
        self.x_ini = 0
        self.y_ini = 0
        self.x_f = 0
        self.y_f = 0

        imatge_but = Button(text='Selecciona una imatge .png o .jpg per crear el teu propi laberint')
        imatge_but.bind(on_press=self.seleccionar_imatge)
        self.add_widget(imatge_but)
        self.imatge_but = imatge_but

        mida_lab = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        mida_lab.add_widget(Label(text='Introdueix la mida del laberint (resolució):', text_size=(self.width, None)))
        mida_lab_input = TextInput()
        mida_lab.add_widget(mida_lab_input)
        mida_lab.add_widget(Button(text='Enviar', on_press=lambda instance: self.comprovar_mida_lab(instance, mida_lab_input.text)))
        self.mida_lab = mida_lab

        ini_lab = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        ini_lab.add_widget(Label(text='Introdueix la posició inicial (x, y):', text_size=(self.width, None)))
        ini_lab_x = TextInput(text='x')
        ini_lab_y = TextInput(text='y')
        ini_lab.add_widget(ini_lab_x)
        ini_lab.add_widget(ini_lab_y)
        ini_lab.add_widget(Button(text='Enviar', on_press=lambda instance: self.comprovar_ini_lab(instance, ini_lab_x.text, ini_lab_y.text)))
        self.ini_lab=ini_lab

        f_lab = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        f_lab.add_widget(Label(text='Introdueix la posició final (x, y):', text_size=(self.width, None)))
        f_lab_x = TextInput(text='x')
        f_lab_y = TextInput(text='y')
        f_lab.add_widget(f_lab_x)
        f_lab.add_widget(f_lab_y)
        f_lab.add_widget(Button(text='Enviar', on_press=lambda instance: self.comprovar_f_lab(instance, f_lab_x.text, f_lab_y.text)))
        self.f_lab = f_lab

    def seleccionar_imatge(self, instance):
        filechooser = FileChooserIconView(filters=['*.png', '*.jpg'])
        filechooser.bind(on_submit=self.seleccionar_fitxer)
        self.add_widget(filechooser)
    
    def seleccionar_fitxer(self, instance, file, touch):
        self.remove_widget(instance)
        self.remove_widget(self.imatge_but)
        self.fitxer = Image.open(file[0])
        self.add_widget(Label(text='Imatge seleccionada: ' + file[0], size_hint_y=0.2))
        self.add_widget(self.mida_lab)

    def comprovar_mida_lab(self, instance, mida):
        try:
            mida = int(mida)
            self.mida=mida
            self.remove_widget(self.mida_lab)
            self.add_widget(Label(text='Mida del laberint: ' + str(mida), size_hint_y=0.2))
            self.add_widget(self.ini_lab)
        except ValueError:
            label = Label(text='Introdueix un nombre enter vàlid', size_hint_y=0.1)
            self.add_widget(label)
            Clock.schedule_once(lambda dt: self.remove_widget(label), 2)
            return
        
    def comprovar_ini_lab(self, instance, x, y):
        try:
            x = int(x)
            y = int(y)
            if x >= self.mida or y >= self.mida:
                label = Label(text='La posició inicial no pot ser fora del laberint', size_hint_y=0.1)
                self.add_widget(label)
                Clock.schedule_once(lambda dt: self.remove_widget(label), 2)
                return
            self.x_ini = x
            self.y_ini = y
            self.remove_widget(self.ini_lab)
            self.add_widget(Label(text='Posició inicial: (' + str(x) + ', ' + str(y) + ')', size_hint_y=0.2))
            self.add_widget(self.f_lab)
        except ValueError:
            label = Label(text='Introdueix un nombre enter vàlid', size_hint_y=0.1)
            self.add_widget(label)
            Clock.schedule_once(lambda dt: self.remove_widget(label), 2)
            return
        
    def comprovar_f_lab(self, instance, x, y):
        try:
            x = int(x)
            y = int(y)
            if x >= self.mida or y >= self.mida:
                label = Label(text='La posició final no pot ser fora del laberint', size_hint_y=0.1)
                self.add_widget(label)
                Clock.schedule_once(lambda dt: self.remove_widget(label), 2)
                return
            self.x_f = x
            self.y_f = y
            self.remove_widget(self.f_lab)
            self.add_widget(Label(text='Posició final: (' + str(x) + ', ' + str(y) + ')', size_hint_y=0.2))
            self.finestra_jugar(instance, self.fitxer, self.mida, self.x_ini, self.y_ini, self.x_f, self.y_f)
        except ValueError:
            label = Label(text='Introdueix un nombre enter vàlid', size_hint_y=0.1)
            self.add_widget(label)
            Clock.schedule_once(lambda dt: self.remove_widget(label), 2)
            return
    
    def finestra_jugar(self, instance, imagen, nivell, x_ini, y_ini, x_f, y_f):
        App.get_running_app().root.clear_widgets()
        App.get_running_app().root.add_widget(Finestra(imagen, nivell, x_ini, y_ini, x_f, y_f))





class Finestra(BoxLayout):
    """Interfase de joc"""
    def __init__(self, imagen, nivell, x_ini, y_ini, x_f, y_f, **kwargs):
        super(Finestra, self).__init__(**kwargs)
        tauler = Tauler(imagen, nivell, x_ini, y_ini, x_f, y_f)
        fletxes = Fletxes(tauler)
        fletxes.size_hint = (0.25, 0.8)
        fletxes.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        tauler.size_hint = (0.7, 0.8)
        tauler.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        self.add_widget(Label(text='', size_hint_x=0.05))
        self.add_widget(fletxes)
        self.add_widget(Label(text='', size_hint_x=0.05))
        self.add_widget(tauler)
        self.add_widget(Label(text='', size_hint_x=0.05))





class Fletxes(BoxLayout):
    
    def __init__(self, tauler, **kwargs):
        super().__init__(**kwargs)

        self.orientation = "vertical"
        self.tauler = tauler

        Window.bind(on_key_down=self.on_key_down)

        esquerra = Button(text='<-')
        esquerra.bind(on_press=self.tauler.on_esquerra)
        dreta = Button(text='->')
        dreta.bind(on_press=self.tauler.on_dreta)
        amunt = Button(text='^')
        amunt.bind(on_press=self.tauler.on_amunt)
        avall = Button(text='')
        avall.bind(on_press=self.tauler.on_avall)

        amuntavall = BoxLayout(orientation='vertical')
        amuntavall.add_widget(amunt)
        amuntavall.add_widget(avall)

        fletxes = BoxLayout(orientation='horizontal', size_hint_y=0.2)
        fletxes.add_widget(esquerra)
        fletxes.add_widget(amuntavall)
        fletxes.add_widget(dreta)
        self.add_widget(fletxes)

        self.add_widget(Label(text='', size_hint_y=0.05))

        controls = BoxLayout(orientation='horizontal', size_hint_y=0.4)
        potencial = BoxLayout(orientation='vertical')
        potencial.add_widget(Label(text='Potencial: ', size_hint_y=0.1))
        valor_V = Slider(min=0.1, max=1, value=0.5, orientation='vertical')
        potencial.add_widget(valor_V)
        etiqueta_valor_V = Label(text='0.5', size_hint_y=0.2)
        valor_V.bind(value=lambda instance, value: self.actualitzar(valor_V, etiqueta_valor_V))
        potencial.add_widget(etiqueta_valor_V)
        controls.add_widget(potencial)
        pas_temps = BoxLayout(orientation='vertical')
        pas_temps.add_widget(Label(text='Pas de temps: ', size_hint_y=0.1))
        valor_t = Slider(min=1, max=100, value=50, step=1, orientation='vertical')
        pas_temps.add_widget(valor_t)
        etiqueta_valor_t = Label(text='50', size_hint_y=0.2)
        valor_t.bind(value=lambda instance, value: self.actualitzar(valor_t, etiqueta_valor_t))
        pas_temps.add_widget(etiqueta_valor_t)
        controls.add_widget(pas_temps)
        self.add_widget(controls)
        self.valor_V = valor_V
        self.valor_t = valor_t

        self.add_widget(Label(text='', size_hint_y=0.05))

        mesura = Button(text='Mesura', size_hint_y=0.1)
        mesura.bind(on_press=self.tauler.mesura)
        self.add_widget(mesura)

        temps = Button(text='Temps', size_hint_y=0.1)
        temps.bind(on_press=self.tauler.temps2)
        self.add_widget(temps)

        inici = Button(text='Inici', size_hint_y=0.1)
        inici.bind(on_press=self.inici)
        self.add_widget(inici)



    def inici(self, instance):
        App.get_running_app().root.clear_widgets()
        App.get_running_app().root.add_widget(Inici())



    def on_key_down(self, window, key, *args):
        if key == 273:
            self.tauler.on_amunt(self)
        if key == 274:
            self.tauler.on_avall(self)
        if key == 275:
            self.tauler.on_dreta(self)
        if key == 276:
            self.tauler.on_esquerra(self)


    
    def actualitzar(self, slider, etiqueta):
        etiqueta.text = "{:.2f}".format(slider.value)
        if slider == self.valor_V:
            self.tauler.actualitzar_potencial(slider.value)
        if slider == self.valor_t:
            self.tauler.actualitzar_pas_temps(slider.value)
        





class Tauler(RelativeLayout):

    def __init__(self, imagen, nivell, x_ini, y_ini, x_f, y_f, **kwargs):
        super().__init__(**kwargs)

        self.mida = 20
        self.funcona = []
        self.nivell = nivell
        self.x_ini = x_ini
        self.y_ini = y_ini
        self.x_f = x_f
        self.y_f = y_f
        self.t = 50

        imagen = imagen.resize((self.nivell,self.nivell))
        imagen = imagen.convert("L")
        self.pixeles = np.array(imagen)
        self.V = np.zeros((self.nivell, self.nivell))
        self.V[self.pixeles < 128 ] = 0.5

        self.func = np.zeros((self.nivell, self.nivell), dtype=np.complex)
        self.func[y_ini, x_ini] = 1
        self.func2 = np.zeros((self.nivell, self.nivell), dtype=np.complex)
        self.func2[y_ini, x_ini] = 1
        self.prob = np.zeros((self.nivell, self.nivell))
        self.prob[y_ini, x_ini] = 1

        self.particula()

    

    def actualitzar_potencial(self, valor):
        for j in range(self.nivell):
            for i in range(self.nivell):
                if self.V[j][i] != 0:
                    self.V[j][i] = valor
        self.particula()

        

    def laberint(self, **kwargs):
        self.parets = []
        self.final = []
        with self.canvas:
            Color(0, 0.2, 0)
            self.fons=Rectangle(pos=(0, 0), size=(self.nivell*self.mida, self.nivell*self.mida))
            for j in range(self.nivell):
                for i in range(self.nivell):
                    if not self.V[j][i] == 0:
                        Color(0, 0.5, 0, self.V[j][i])
                        self.parets.append([Rectangle(pos=(i*self.mida, j*self.mida), size=(self.mida,self.mida)), i, j])
            Color(1, 0, 0)
            self.final.append([Rectangle(pos=(self.x_f*self.mida, self.y_f*self.mida), size=(self.mida,self.mida)), self.x_f, self.y_f])
        self.on_size()



    def particula(self):
        self.canvas.clear()
        self.laberint()
        ppi_x = self.width/2 - self.nivell*self.mida/2
        ppi_y = self.height/2 - self.nivell*self.mida/2
        max_prob = 0
        for j in range(self.nivell):
            for i in range(self.nivell):
                if self.V[j][i] == 0:
                    if self.prob[j][i] > max_prob:
                        max_prob = self.prob[j][i]
        for j in range(self.nivell):
            for i in range(self.nivell):
                if not self.prob[j][i] == 0:
                    if self.V[j][i] == 0:
                        with self.canvas:
                            Color(1, 1, 1, (1-0.03)/(max_prob)*self.prob[j][i]+0.03)
                            self.funcona.append([Rectangle(pos=(i*self.mida+ppi_x, j*self.mida+ppi_y), size=(self.mida,self.mida)), i, j])



    def on_size(self, *args):
        self.mida = min(self.width/self.nivell, self.height/self.nivell)
        ppi_x = self.width/2 - self.nivell*self.mida/2
        ppi_y = self.height/2 - self.nivell*self.mida/2
        for i in self.parets:
            i[0].size = (self.mida, self.mida)
            i[0].pos = (i[1]*self.mida+ppi_x, i[2]*self.mida+ppi_y)
        for i in self.funcona:
            i[0].size = (self.mida, self.mida)
            i[0].pos = (i[1]*self.mida+ppi_x, i[2]*self.mida+ppi_y)
        self.fons.size = (self.nivell*self.mida, self.nivell*self.mida)
        self.fons.pos = (ppi_x, ppi_y)
        for i in self.final:
            i[0].size = (self.mida, self.mida)
            i[0].pos = (i[1]*self.mida+ppi_x, i[2]*self.mida+ppi_y)



    def on_dreta(self, instance):
        for j in range(self.nivell):
            for i in range(self.nivell):
                if i < self.nivell-1:
                    if self.V[j][i+1] == 0:
                        self.func2[j][i+1] += self.func[j][i]
                    else:
                        self.func2[j][i] += self.func[j][i]
                if i == self.nivell-1:
                    self.func2[j][i] += self.func[j][i]
        for j in range(self.nivell):
            for i in range(self.nivell):
                self.func[j][i] = self.func2[j][i]
                self.func2[j][i]=0
        self.temps()

    def on_esquerra(self, instance):
        for j in range(self.nivell):
            for i in range(self.nivell):
                if i > 0:
                    if self.V[j][i-1] == 0:
                        self.func2[j][i-1] += self.func[j][i]
                    else:  
                        self.func2[j][i] += self.func[j][i]
                if i == 0:
                    self.func2[j][i] += self.func[j][i]
        for j in range(self.nivell):
            for i in range(self.nivell):
                self.func[j][i] = self.func2[j][i]
                self.func2[j][i]=0
        self.temps()

    def on_amunt(self, instance):
        for j in range(self.nivell):
            for i in range(self.nivell):
                if j < self.nivell-1:
                    if self.V[j+1][i] == 0:
                        self.func2[j+1][i] += self.func[j][i]
                    else:
                        self.func2[j][i] += self.func[j][i]
                if j == self.nivell-1:
                    self.func2[j][i] += self.func[j][i]
        for j in range(self.nivell):
            for i in range(self.nivell):
                self.func[j][i] = self.func2[j][i]
                self.func2[j][i] = 0
        self.temps()
    
    def on_avall(self, temps):
        for j in range(self.nivell):
            for i in range(self.nivell):
                if j > 0:
                    if self.V[j-1][i] == 0:
                        self.func2[j-1][i] += self.func[j][i]
                    else:    
                        self.func2[j][i] += self.func[j][i]
                if j == 0:
                    self.func2[j][i] += self.func[j][i]
        for j in range(self.nivell):
            for i in range(self.nivell):
                self.func[j][i] = self.func2[j][i]
                self.func2[j][i]=0
        self.temps()



    def actualitzar_pas_temps(self, valor):
        self.t = valor
    


    def temps(self):

        hbar = 1
        m = hbar**2/(2*61.04212604) # massa de l'electró
        delta = self.mida

        coef = (self.t*hbar)/(1j*2*m*delta**2)

        for j in range(self.nivell):
            for i in range(self.nivell):
                self.func2[j][i] = self.func[j][i] * ( 1 + self.V[j][i]*self.t/(1j*hbar) + (2*self.t*hbar)/(1j*m*delta**2) ) 
                if j<self.nivell-1:
                    self.func2[j][i] -= coef * self.func[j+1][i]
                if j>0:
                    self.func2[j][i] -= coef * self.func[j-1][i]
                if i<self.nivell-1:
                    self.func2[j][i] -= coef * self.func[j][i+1]
                if i>0:
                    self.func2[j][i] -= coef * self.func[j][i-1]

        suma=0
        for j in range(self.nivell):
            for i in range(self.nivell):
                self.func[j][i] = self.func2[j][i]
                self.prob[j][i] = (abs(self.func2[j][i]))**2
                suma += self.prob[j][i]
        for j in range(self.nivell):
            for i in range(self.nivell):
                self.prob[j][i] = self.prob[j][i] / suma
                self.func[j][i] = self.func[j][i] / suma
                self.func2[j][i] = 0

        self.particula()


    def temps2(self, instance):
        self.temps()
    
    def mesura(self, instance):
        # Aplanar las matrices lab y prob en listas unidimensionales, rastreando las posiciones
        func_flat = [(j, i) for j, sublist in enumerate(self.func) for i, item in enumerate(sublist)]
        prob_flat = [item for sublist in self.prob for item in sublist]
        # Seleccionar una posición de lab con la probabilidad correspondiente en prob
        selected_position = random.choices(func_flat, weights=prob_flat, k=1)[0]
        while self.V[selected_position[0]][selected_position[1]] != 0:
            selected_position = random.choices(func_flat, weights=prob_flat, k=1)[0]
        # Actualitzar la funció d'ona
        self.func = [[0 for j in range(self.nivell)] for i in range(self.nivell)]
        self.func[selected_position[0]][selected_position[1]] = 1
        self.prob = [[0 for j in range(self.nivell)] for i in range(self.nivell)]
        self.prob[selected_position[0]][selected_position[1]] = 1
        self.particula()
        #Mirar si ha acabat el joc
        for i in self.final:
            if i[1] == selected_position[1] and i[2] == selected_position[0]:
                self.final_joc()
    
    def final_joc(self):
        popup = Popup(title='Felicitats!',
              content=Label(text='Has aconseguit sortir del laberint!'),
              size_hint=(0.5, 0.5))
        popup.open()
        App.get_running_app().root.clear_widgets()
        App.get_running_app().root.add_widget(Inici())





LaberintApp().run()
