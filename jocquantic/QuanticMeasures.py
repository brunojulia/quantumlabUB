#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs Quantic Measures app. Contains every class used there as well. Uses the 
kivy file 'QuanticMEasures.kv' to build the app interface.

----------------
Script structure
----------------
-Imports
-Classes
-App run

------------
Classes list
------------
QuanticMeasuresApp: App's child
    The application class. It's function run() is called when running the 
    script.
MyScreenManager: ScreenManager's child
    Manages screens set up and has access to every screen class. 
    QuanticMeasuresApp is built with this class.
StartingScreen: Screen's child
    One out of three screens. The app opens with this screen. Contains two 
    popups.
InfoPopup: Popup's child
    Starting screen popup containing information about the game.
PhysicsPopup: Popup's child
    Starting screen popup containing information about the physics of the game.
GamingScreen: Screen's child
    One out of three screens. The gaming part of the application is here.
IllustratingScreen: Screen's child
    One out of three screens. Demos are shown here.
    
-----------------------   
Making an app with kivy
-----------------------
With kivy's class App, and calling its method run() you create the most basic 
app. 

In order to define custom functions for the app to use, an App's child needs
to be created. Say some custom class, CustomExample, contains these functions. 
Then a class inheriting from App will override the method build(), returning 
the CustomExample class (see QuanticMeasureApp.build()). This App's child run()
method is then called. The app now has access to every CustomExample class 
function, and a button for example could use any of them.

In order to construct the interface, a kivy file or a builder string are 
needed. In this case a kivy file. There, the interface (buttons, layouts,
images, sliders...) are created using kivy syntax.
                                                
    Kivy syntax:
        For every class the app is going to directly use (a button calling it
        directly this is, for example), a widget tree is constructed. This tree
        basically is a list of widgets, each one with id, properties, events 
        bound to functions... By indentation and using Layout widgets, the 
        final set up is defined. See BoxLayout documentation for a clear 
        example:
           https://kivy.org/doc/stable/api-kivy.uix.boxlayout.html#
           
How to relate App's child class in the python file to the kivy file with 
the interface for the app to use? There are multiple ways. The one used here
uses naming. If the kivy file is named Example.kv, the App's child class must 
be named ExampleApp. Then, ExampleApp().run() will use Example.kv script as
interface, and not others.
        
--------------
About this app
--------------
This app is about popularizing quantum mechanics, as part of the project 
QuantumlabUB. In this case, the wave function object and the randomness of 
measuring are to be popularized. 

An specific problem is used: the time evolution of the wave function of a 
particle, inside box with a certain potential inside (1D). 

The way popularizing is done is by building a game around this evolution 
problem (interacting with it by measures) and by showing certain illustrative 
cases. Hence the screens structure: starting, gaming and illustrative screens.

The starting screen is the initial screen when opening the app. Has to 
information popups: about how to play the game and about the physics problem
(thought with the complementing explanation of a person when showing the game).

The gaming screen. Basically, by looking at the evolution of the wave function,
a measure in the position has to be made in a precise moment, with the goal in 
mind that the resulting measure (a position) has to be in a given zone (marked
with colors, red and green). Depending on whether the goal is achieved or not,
the player looses a life.

In the illustrating screen, there's just the evolution (no lives system).
Some buttons are implemented so the player can move through different demos.

----------------
General comments
----------------
In this script almost every function is designed more as a fortran subroutine.
In the sense that they don't take an input as variables and return a certain 
output. Mostly they carry on certain actions, like transforming certain global 
variable from the class, or acting on the events flow.

Docstrings on functions should have been in the way: 'do' that, instead of the 
actual docstrings, 'does' that.
 
-------------------------------------------------------------------------------
Created on Thu Dec  5 20:13:52 2019 (most of the code was copied from an older
script, which was created on Sat Apr 27 17:29:25 2019. Same author.)
@author: Manu Canals
"""
###############################################################################
#                               IMPORTS                                       #
###############################################################################
#KIVY'S IMPORTS
#--------------
from kivy.app import App 
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen 
from kivy.uix.screenmanager import FadeTransition, SlideTransition   
from kivy.uix.popup import Popup
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.core.text import LabelBase
from kivy.core.window import Window
#OTHER GENERAL IMPORTS
#–--------------------
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
from matplotlib import gridspec
from scipy import linalg as spLA 
#import timeit as ti 
#FONT 'IMPORTS' 
#-----------------
#8_bit used on every text. VT323 takes care of accents.
LabelBase.register(name = '8_bit_madness', 
                   fn_regular = 'Files/8_bit_madness-regular.ttf')
LabelBase.register(name = 'VT323', 
                   fn_regular = 'Files/VT323-regular.ttf')

###############################################################################
#                                CLASSES                                      #
###############################################################################
class QuanticMeasuresApp(App):
    """'THE' app class.
    
    Its parent, App, contains the fundamental for creating the app. 
    
    Functions
    ---------
    run():
        Inherited from App. Fires the app.
    build():
        Passes custom functions to the app.
    """
    def build(self):
        """Passes an instance of MyScreenManager to the app.
        
        The function build() is inherited from App as well, but gets 
        overridden so the instance can be passed. 
        
        Gives the app access to MyScreenManager methods. Called from run().
        """
        self.title = 'Quantic Measures'
        return MyScreenManager()

class MyScreenManager(ScreenManager):
    """Manages and has access to the screens.
    
    In order to create the Screens, a manager has to exist. 
    
    Functions
    ---------
    __init__():
        Initiates the manager (it and its screens.
    get_screen(screen_name):
        Returns the widget (instance) of the given screen.
    """
    
    def  __init__(self, **kwargs):
        """Initiates the manager and the screens.
        
        Called whenever an instance of MyScreenManager is created.
        
        Calls ScreenManager __init__ method, which calls in turn every screen's
        __init__ function. This creates all the widgets defined in kivy file,
        and importantly, creates their ids. So only after this __init__, the 
        ids can be referred to. Hence, why pseudo_init function had to be 
        defined.
        """
        super(MyScreenManager, self).__init__(**kwargs)
        self.get_screen('gaming').gpseudo_init()
        self.get_screen('illustrating').ipseudo_init()

class StartingScreen(Screen):
    """Initial screen when the app is opened.
    
    Its managed by MyScreenManager, here called manager.
    
    No animation takes place here.
    
    Layout
    ------
    -BoxLayout: 
    (*every class can only have one widget, so every widget is in a BoxLayout)
        *Title image
        *BoxLayout:
            +Game button: transitions to gaming screen
            +Demos button: transitions to illustrating screen
            +Info popup: info about the game
            +Physics popup: info about the physics of the game
        *UB logo image
        
    Functions
    ---------
    __init__:
        Calls the Screen __init__ function.
    transition_SI:
        Does the transition to the illustrating screen, including calls to 
        other methods apart from moving to that screen.
    transition_SG:
        Does the transition to the gaming screen, including calls to other 
        methods apart from moving to that screen.
    
    """
    def __init__ (self, **kwargs):
        """Initiates the screen by calling Screen's __init__.
        
        Called from ScreenManager __init__.
        """
        super(StartingScreen, self).__init__(**kwargs)
        
    def transition_SI(self):
        """Screen transition: from 'starting' to 'illustrating'.
        
        Includes:
            Firing illustrating screen animation loop.
            Requesting keyboard for illustrating screen.
            Moving to illustrating.
        """
        iscreen = self.manager.get_screen('illustrating')
        iscreen.i_schedule_fired()
        iscreen.request_KB()
        self.manager.transition = FadeTransition()
        self.manager.current = 'illustrating'

    def transition_SG(self):
        """Screen transition: from 'starting' to 'gaming'.
        
        Includes:
            Firing gaming screen animation loop.
            Requesting keyboard for gaming screen.
            Moving to gaming.
        """
        gscreen = self.manager.get_screen('gaming')
        gscreen.g_schedule_fired()
        gscreen.request_KB()
        self.manager.transition = FadeTransition()
        self.manager.current = 'gaming'
        
class InfoPopup(Popup):
    """Information about the game popup.
    
    Its text is read from a file. This reading only happens once when starting
    the app.
    
    Layout
    -----
    -ScrollView:
        *BoxLayout:
             +Intro label
             +Game screenshot, image
             +Game elements label
             +Author label
    
    Functions
    ---------
    __init__:
        Initiates the popup.
    """
    with open('Files/info_text.txt', 'r') as t:
            lines = t.readlines()
    
    intro_text = lines[0] + '\n' +lines[1] +'\n' + lines[2] + '\n' + lines[3]
    gamele_text = '\n' + lines[4] + '\n' + lines[5] + '\n' + lines[6] + '\n' +\
                         lines[7] + '\n' + lines[8] + '\n' + lines[9] + '\n' +\
                         lines[10] + '\n' + lines[11] + '\n' + lines[12] + \
                         '\n' + lines[13]
    
    def __init__(self):
        """Initiates the popup.
        
        Called every time the popup is opened. Assigns the read text to the 
        label variables.
        """
        super(Popup, self).__init__()
        self.intro.text = self.intro_text
        self.gamele.text = self.gamele_text
        
class PhysicsPopup(Popup):
    """Information about the physics of the game popup.
    
    Its text is read from a file. This reading only happens once when starting
    the app.
    
    Layout
    -----
    -ScrollView:
        *Label
    
    Functions
    ---------
    __init__:
        Initiates the popup.
    """
    with open('Files/physics_text.txt', 'r') as t:
            lines = t.readlines()
    
    text = lines[0] + '\n' +lines[1] +'\n' + lines[2] + '\n' + lines[3] + '\n'\
         + lines[4] + '\n' +lines[5] +'\n' + lines[6] + '\n' + lines[7] + '\n'\
         + lines[8]
    
    def __init__(self):
        """Initiates the popup.
        
        Called every time the popup is opened. Assigns the read text to the 
        label variables.
        """
        super(Popup, self).__init__()  
        self.phypop.text = self.text

class GamingScreen(Screen):
    """Screen containing the game. Explicitly stops everything when it's not 
    the current screen.
    
    Managed by MyScreenManager, here called manager.
    
    This class consists of three main blocks:
        
        - COMPUTATION of a wave function's evolution on different potentials.
        - PLOTTING of this evolution.
        - GAME. A game is build on top of this problem.
    
    All functions will be organized with this structure in mind. Nevertheless,
    the core of this class are animation and events flow, managed only
    by a few functions – they actually use the functions from the said main 
    blocks.
    
        - FLOWING functions, manage the events flow.
    
    Layout
    ------
    -BoxLayout: (containing the rest of the widgets)
        *BoxLayout: (lives, plot, buttons)
            +Level label
            +BoxLayout: (lives)
                ·lives x10
            +BoxLayout: (plot)
                ·Plot box
            +BoxLayout: (buttons)
                ·Pause
                ·BoxLayout: (jokers, help)
                    #BoxLayout: (help)
                        --Help label
                        --Help button
                    #Joker label
                    #BoxLayout: (jokers)
                        --jokers x3
        *BoxLayout: (go demos, starting)
            +Go demos button
            +Go start button
    
    Functions
    ---------
    (init)
        __init__:
            Initiates the screen.
        gpseudo_init:
            First draw, first problem solve and declaration of some variables.
    (flowing)
        g_schdeule_fired:
            Starts the looping on this screen.
        g_schedule_cancel:
            Stops the looping on this screen.
        plotpsiev:
            Animation function. Repeatedly called. Draws every frame.
        meausure:
            Picks one random position with the instant wave function, checks if
            the level is passed or not, and restarts psi.
        skip_lvl:
            Measures but always passing the level. Used via jokers.
        restart:
            When the game is over, resets to initial parameters.
        pause:
            Changes the paused state effectively pausing or playing the game.
    (computation)
        eigenparam:
            Given a potential, builds the hamiltonian and computes its eigen
            basis and eigen values.
        comp:
            In the eigen basis, computes the components if the current psi.
        psi_init:
            Creates a gaussian wave function in a given position.
    (plotting)
        fill_bkg:
            Paints the color zones of the plot, the area under the potential
            and the area under psi as well.
        fill_zones:
            Paints the zones plot (under main animated plot) and builds 
            redzone.
        measure_arrow:
            Draws arrows indicators of the result of a measure.
        fake_axis:
            Draws arrows and annotations as fake axis.
    (game)
        live_resources:
            Updates lives images when loosing/gaining them.
        _keyboard_closed:
            Actions taken when the keyboard is released: unbind and setting 
            keyboard to None.
        request_KB:
            Bounded to the whole plot box, binds keyboard and optionally 
            restarts the game.
        _on_keyboard_down:
            Functions that keyboard is actually bound to: keyboard triggers
            measure.
        helping_mode:
            Switch helping state. If helping, measure results not random, 
            always the maximum probability position.
        transition_GS:
            Does the transition to the starting screen, including calls to 
            other methods apart from moving to that screen.
        transition_GI:
            Does the transition to the illustrative screen, including calls to 
            other methods apart from moving to that screen.
        generate_lvl_settings:
            Creates the potential and the position of the green zone.
        lvl_up:
            Changes the speed and green zone width after leveling up.
    """
    
    ###########################################################################
    #                            INIT FUNCTIONS                               #
    ###########################################################################
#    __init__:
#        Initiates the screen.
#    gpseudo_init:
#        First draw, first problem solve and declaration of some variables.
    
    def  __init__(self, **kwargs):
        """Initiates the screen by calling Screen's __init__.
        
        Called from My ScreenManager __init__. When called, kivy widgets still
        don't exist, so no references to any ids can be made, hence the second
        init function: 'pseudo_init'.
        """
        super(GamingScreen, self).__init__(**kwargs)#Inits also the superclass

    def gpseudo_init(self):    
        """Initiates the game itself, by doing the first solving, first drawing
        and setting up the game.
        
        Sets everything ready for the loop to start calling plotpsiev. Detailed
        explanation along the code.
        
        Called from MyScreenManager init after its super __init__, when ids 
        already exists.
        """
        #======================== FIRST SOLVING ===========================                           
#        Solving this problem has two parts: finding the EIGENBASIS and 
#        eigenvalues of the given hamiltonian (with a given potential), and
#        finding the COMPONENTS of the initial psi in this basis. 
        
            #SOLVING. 1ST PART
                                             
        #UNITS
        self.unit_time = 'fs' 
        self.unit_energy = 'eV'
        self.unit_long = '$\AA$'
        self.unit_probxlong = '$\AA^{-1}$'
        
        #CONSTANTS
        self.hbar = 0.6582   #In these general units
        self.m_elec = 0.1316 #Its the m factor explained in eigenparam function
        self.m = self.m_elec #The name 'm' is the one used inside of eigenparam
        self.dirac_sigma = 0.6

        #DISCRETIZATION
        self.a = -10.   
        self.b = 10.        
        self.N = 800
        self.deltax = (self.b - self.a)/float(self.N)
        self.mesh = np.linspace(self.a, self.b, self.N + 1)
        
        #POTENTIAL INITIAL SETTINGS
        self.mu0 = 1 
        self.generate_lvl_settings()
        
        #EIGENBASIS
        self.eigenparam()
       
            #SOLVING. 2ND PART
        
        #PSI INITIAL SETTINGS (after every measure)
        #self.p0 = 0. 
        self.sigma0 = self.dirac_sigma
        self.psi_init()
        
        #COMPONENTS & ENERGY
        self.comp()
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)

        #=========================== FIRST DRAW ===============================  
#        Matplotlib plotting, its figure is going to be passed to kivy's 
#        renderer (sort of the canvas) via FigureCanvasKivyAgg. The canvas then 
#        its added as a widget to the plot box (see Layout in class docstring).
#        
#        This box is the one bound to request_KB and restarting.
#        
#        The whole plot consists of three subplots (in a vertical grid):
#            -bkg_twin:
#                Main plot. Contains color zones, under the curve fills for 
#                the potential and 'psi' (later explained) and indicator 
#                arrows.  Has a sharing x axis twin:
#                    *pot_twin: Contains potential, energy and fading images.
#            -zones:
#                Under the previous plot. Contains just the color zones.
#            -visuax:
#                Under the previous plot. Visual image of the wave function
#                by a gray map.
#        
#        For the sake of faster frame rate, no axis are going to be drawn. 
#        Instead fake ones are drawn: arrow and annotations ( fake_axis() ).
        
        
        #COLORS
        self.zonecol_red = '#AA3939'
        self.zonecol_green = '#7B9F35'
        self.potcol = '#226666'
        self.cmap_name = 'gray'
        self.energycol = '#AA8439'
        self.b_arrow_color = '#C0C0C0'
        self.u_arrow_color = '#582A72'
        
        
        #LIMITS
        self.pot_tlim = 50
        self.pot_blim = 0
        self.psi_tlim = 1.7
        self.psi_blim = 0
        Dpot = self.pot_tlim - self.pot_blim
        Dpsi = self.psi_tlim - self.psi_blim
        self.dcoord_factor = Dpot / Dpsi
        
        
        #FIGURE
        self.main_fig = plt.figure()
        self.main_fig.patch.set_facecolor('black') 
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig) #Passed to kv
        self.box1.add_widget(self.main_canvas)
        self.main_canvas.bind(on_touch_up = self.request_KB)
        self.gs = gridspec.GridSpec(3, 1, height_ratios=[7, 1, 1], 
                          hspace=0.1, bottom = 0.05, top = 0.90) #Subplots grid
        
        
        #BACKGROUND
        self.bkg_twin = plt.subplot(self.gs[0])
        self.bkg_twin.axis([self.a, self.b, self.psi_blim, self.psi_tlim])
        self.bkg_twin.axis('off')
        self.bkg_twin.set_facecolor('black')
            #Color zones and fills
        self.init_zones_width = 4
        self.zones_width = self.init_zones_width
        self.fill_bkg()
            #Arrows (first drawn transparent just to create the instance)
        self.b_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.u_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        
        
        #POTENTIAL
        self.pot_twin = self.bkg_twin.twinx()
        self.pot_twin.axis([self.a, self.b, self.pot_blim, self.pot_tlim])
        self.pot_twin.axis('off')
        self.pot_twin.set_facecolor('black')
        self.pot_data, = self.pot_twin.plot(self.mesh, self.potential, 
                                            color = self.potcol)
        self.E_data, = self.pot_twin.plot(self.mesh, 
                                         np.zeros_like(self.mesh)+self.energy, 
                                         color = self.energycol, ls = '--',
                                         lw = 2)
            #Fading images
        self.gameover_imgdata=mpimg.imread('Images/gameover_img.png',0) #G.OVER
        self.kivy_skull_imgdata = mpimg.imread('Images/skull_img.png', 0)
        self.init_alpha = 0.7
        self.skull_fading = None 
        self.lvlup_imgdata = mpimg.imread('Images/lvl_up.png')         #LVLUP
        self.lvlup_fading = None
        #'Image'_fading can be either None or an instance of imshow. If None:
        #no set_alpha will occur, else, setting alpha and removing will be 
        #done.
        
        
        #AXIS
        self.fake_axis()
        
        
        #ZONES
        self.zones = plt.subplot(self.gs[1])
        self.zones.axis([self.a, self.b, 0, 1])
        self.zones.axis('off')
        self.fill_zones()
        
        
        #VISUAL
        self.visuax = plt.subplot(self.gs[2])
        self.num_visu = len(self.mesh) #Can't be greater than the # of indices
        self.inter_visu = 'gaussian'
        self.visuax.axis('off')
        step = int(len(self.psi)/self.num_visu) #num_visu points in gray map
        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
                 aspect='auto', interpolation = self.inter_visu, 
                 cmap = self.cmap_name)
        
        
        #FINALLY, ACTUAL FIRST DRAW
        #This 'tight' needs to be at the end so it considers all objects drawn
        self.main_fig.tight_layout() #Fills all available space
        self.main_canvas.draw()
        
        #====================== GAME INITIAL SETTINGS ========================= 
        #GAME VARIABLES
        self.kivy_heart_img = 'Images/heart_img.png'                     #HEART
        self.kivy_skull_img = 'Images/skull_img.png'                     #SKULL
        self.max_lives = 10 #If changed, kv file needs to be changed as well
        self.lives = self.max_lives 
        self.lives_sources() 
        self.helping = False
        self.lvl = 1 
        
        
        #KEYBOARD
        #request_keyboard returns an instance that represents the events on 
        #the keyboard. It can give two events(which can be bound to functions).
        #Furthermore, each event comes with some input arguments:
        #on_key_down    --      keycode, text, modifiers
        #on_key_up      --      keycode
        #Using the functions bind and unbind on this instance one can bind or 
        #unbind the instance event with a callback/function AND passing the
        #above arguments into the function.
        #When using request_keyboard a callback/function must be given. It is 
        #going to be called when releasing/closing the keyboard (shutting the 
        #app, or reassigning the keyboard with another request_keyboard). It 
        #usually unbinds everything bound.
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down = self._on_keyboard_down)
   
    
        #TIME
        #Time in this animation is handled by setting a dt and adding it to 
        #the time every frame. Then, factors in that addition can be 
        #introduced.
        self.plt_time = 0.
        self.plt_dt = 1./30.
        self.init_vel = 8
        self.plt_vel_factor = self.init_vel #Factor in dt
        self.pause_state = True #Begins paused
        
    ###########################################################################
    #                          FLOWING FUNCTIONS                              #
    ###########################################################################
#    g_schdeule_fired:
#        Starts the looping on this screen.
#    g_schedule_cancel:
#        Stops the looping on this screen.
#    plotpsiev:
#        Animation function. Repeatedly called. Draws every frame.
#    meausure:
#        Picks one random position with the instant wave function, checks if
#        the level is passed or not, and restarts psi.
#    skip_lvl:
#        Measures but always passing the level. Used via jokers.
#    restart:
#        When the game is over, resets to initial parameters.
#    pause:
#        Changes the paused state effectively pausing or playing the game.
    
    
    def g_schedule_fired(self):
        """Starts the looping on gaming screen.
        
        Called when transitioning between screens.
        
        schdeule_interval orders kivy to try calling the given callback 
        repeatedly with the given dt. If can't be done in the given dt, does it
        the faster it can any way.
        
        Separated with an individual function for variable referencing reasons.
        """
        self.schedule = Clock.schedule_interval(self.plotpsiev, 1/30)
        
        
    def g_schedule_cancel(self):
        """Stops the looping on gaming screen.
        
        Called when transitioning between screens.
        
        Separated with an individual function for variable referencing reasons.
        """
        self.schedule.cancel()
        
        
    def plotpsiev(self, dt):
        """Computes psi(t), update zones and fading.
        
        If playing: updates psi, bkg_twin and fading images alpha, 
        and draws. If paused: draws anyway but no updating occur.
        
        Called from loop, started by schedule_interval.
        
        Updates by doing a time step, computing the new psi (given the 
        components and the eigen values), and refilling bkg_twin and reshowing
        visuax image. Fading images have their alpha set as well.
        
        Has to have dt as argument. Is the real time between frames.
        """
        if not self.pause_state:
            #TIME STEP
            self.plt_time += self.plt_vel_factor*self.plt_dt #Time step          
            t = self.plt_time
            
            
            #COMPUTE PSIEV(t). 
            #We do it with two steps (given t).
                #_1_. 
                #Column vector containing the product between component and 
                #exponential factor. 1st build array ary and then col. matrix 
                #'mtx'
            ary_compexp = self.compo * \
                        np.exp(np.complex(0.,-1.)*self.evals*t/(50*self.hbar))
            mtx_compexp = np.matrix(np.reshape(ary_compexp, (self.N + 1,1)))
                #_2_. 
                #Psi(t)
            col_psi = self.evect * mtx_compexp #Matrix product (faster)
            self.psi = np.array(np.reshape(col_psi, self.N + 1))[0]
            self.psi2 = np.abs(self.psi)**2
        
        
            #ANIMATION (BKG & VISU)
            self.fill_bkg()
            self.visu_im.remove()
            step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
            self.visu_im = self.visuax.imshow([self.psi2[::step]], 
                 aspect='auto', interpolation = self.inter_visu, 
                 cmap = self.cmap_name)
            
            
            #FADING IMAGES
            self.b_arrow.set_alpha(self.init_alpha*np.exp(-t/10))
            self.u_arrow.set_alpha(self.init_alpha*np.exp(-t/10))
            curr_alpha = self.init_alpha*np.exp(-t/2)
            if self.skull_fading != None:
                self.skull_fading.set_alpha(curr_alpha)
            if self.lvlup_fading != None:
                self.lvlup_fading.set_alpha(curr_alpha)
                
            
        #DRAW 
        #(keeps drawing even if there hasn't been an update)
        self.main_canvas.draw()
        
        
    def measure(self):
        """Pick a random position, checks if its in redzone and initiates psi.
        
        Called from the keyboard (space).
        
        Picks new position and initiates psi again there. In between, the zone
        is checked looking if the new position is in the redzone.
        
        If OUT, fading image pops up, a life is lost and checks if the game is 
        over.
        If IN, fading image pops up and the level is passed (and a new one
        generated).
        """
        #NEW MU0
        prob = self.deltax*self.psi2 #Get instant probability
        prob /= sum(prob)
        self.mu0 = np.random.choice(self.mesh, p=prob) #Pick new random mu0
        if self.helping:
            self.mu0 = self.mesh[np.argmax(prob)]
        self.measure_arrow()
        self.plt_time = 0. #Reset time 
        self.sigma0 = self.dirac_sigma #New sigma
        
           
        #CHECKING NEW POSITION
        if self.mu0 in self.redzone:
            #IMAGE POPUPS
            if self.skull_fading != None: #Previously exists an skull
                self.skull_fading.remove()
            self.skull_fading = self.pot_twin.imshow(self.kivy_skull_imgdata,
                                                 aspect = 'auto',
                                                 extent = [-2.5, 2.5, 10, 40],
                                                 alpha = self.init_alpha)
            if self.lvlup_fading != None: #If missed, remove prev lvlup
                self.lvlup_fading.remove()
                self.lvlup_fading = None
                
                
            #LOOSE LIFE
            self.lives -= 1
            self.lives_sources()
            
            
            #GAME OVER
            if self.lives <= 0: 
                if self.skull_fading != None: #When gaming over, remove skull
                    self.skull_fading.remove()
                    self.skull_fading = None
                self.pause_state = True
                self.pause_btn.text = 'PLAY'
                self.GMO_img = self.pot_twin.imshow(self.gameover_imgdata, 
                                  aspect = 'auto', extent = [-7.5, 7.5, 0, 40])
                self._keyboard.release()
                self.joker1.disabled = True
                self.joker2.disabled = True
                self.joker3.disabled = True
            
            
        else: 
            #IMAGE POPUPS
            if self.lvlup_fading != None: #Prev exist an skull
                self.lvlup_fading.remove()
            self.lvlup_fading = self.pot_twin.imshow(self.lvlup_imgdata,
                                                 aspect = 'auto',
                                                 extent = [-3.5, 3.5, 20, 30],
                                                 alpha = self.init_alpha)
            if self.skull_fading != None: #If lvl passed, remove prev skull
                self.skull_fading.remove()
                self.skull_fading = None
                
                
            #NEW LVL
            self.lvl += 1 #Read new lvl
            self.label_lvl.text = 'LEVEL ' + str(self.lvl)
            self.lvl_up() #Width and speed changes
            self.generate_lvl_settings()
            self.fill_zones() #It could go inside generat_lvl_settings()
            self.pot_data.set_data(self.mesh, self.potential)
            self.eigenparam()
                
            
        #NEW PSI
        self.psi_init()
        self.comp()
        self.fill_bkg()
        self.visu_im.remove()
        step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
             aspect='auto', interpolation = self.inter_visu, 
             cmap = self.cmap_name)
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)
        self.E_data.set_data(self.mesh, self.energy)
        
        
    def skip_lvl(self):
        """Skips the current level.
        
        Does exactly what measure does, but always passes the level (no 
        checking).
        
        In the game will be used as jokers (called pressing those).
        """
        #NEW MU0
        prob = self.deltax*self.psi2 #Get instant probability
        prob /= sum(prob)
        self.mu0 = np.random.choice(self.mesh, p=prob) #Pick new random mu0
        if self.helping:
            self.mu0 = self.mesh[np.argmax(prob)]
        self.measure_arrow()
        self.plt_time = 0. #Reset time 
        self.sigma0 = self.dirac_sigma #New sigma
        
        
        #IMAGE POPUPS
        if self.skull_fading != None:
            self.skull_fading.remove()
            self.skull_fading = None
        if self.lvlup_fading != None:
            self.lvlup_fading.remove()
            self.lvlup_fading = None
            
        
        #NEW LVL        
        self.lvl += 1
        self.label_lvl.text = 'LEVEL ' + str(self.lvl)
        self.lvl_up()
        self.generate_lvl_settings()
        self.fill_zones()
        self.pot_data.set_data(self.mesh, self.potential)
        self.eigenparam()
        
        
        #NEW PSI
        self.psi_init()       
        self.comp()
        self.fill_bkg()
        self.visu_im.remove()
        step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
             aspect='auto', interpolation = self.inter_visu, 
             cmap = self.cmap_name)
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)
        self.E_data.set_data(self.mesh, self.energy)
        
        
    #RESTART
    def restart(self):
        """Sets the game to the beggining and pauses.
        
        Called from clicking at the plot box after the game is over (bound to
        the entire box).
        
        Resetting everything by removing images and setting variables to 
        initial values. Then, a new level is created (level 1) and the 
        corresponding psi initiated again.
        """
        #REMOVING AND RESETTING
        self.GMO_img.remove()
        self.b_arrow.remove()
        self.u_arrow.remove()
        self.b_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.u_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.lvl = 1
        self.label_lvl.text = 'LEVEL ' + str(self.lvl)
        self.lives = self.max_lives
        self.lives_sources()
        self.zones_width = self.init_zones_width
        self.plt_vel_factor = self.init_vel
        self.joker1.disabled = False
        self.joker2.disabled = False
        self.joker3.disabled = False
        
        self.pause_state = True
        self.pause_btn.text = 'PLAY'
        
        
        #NEW LVL
        self.generate_lvl_settings()
        self.fill_zones()
        self.pot_data.set_data(self.mesh, self.potential)
        self.eigenparam()


        #NEW PSI
        self.psi_init()
        self.comp()
        self.fill_bkg()
        self.visu_im.remove()
        step = int(len(self.psi)/self.num_visu)
        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
             aspect='auto', interpolation = self.inter_visu, 
             cmap = self.cmap_name)
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)
        self.E_data.set_data(self.mesh, self.energy)
        
        
    def pause(self):
        """Changes the pause state from true to false and viceversa.
        
        Called from the button directly.
        """
        if self.pause_state == True: 
            self.pause_state = False
            self.pause_btn.text = 'PAUSE'
        else:
            self.pause_state = True 
            self.pause_btn.text = 'PLAY'
        
    ###########################################################################
    #                            COMPUTING FUNCTIONS                          #
    ###########################################################################
#    eigenparam:
#        Given a potential, builds the hamiltonian and computes its eigen
#        basis and eigen values.
#    comp:
#        In the eigen basis, computes the components if the current psi.
#    psi_init:
#        Creates a gaussian wave function in a given position.
 
    def eigenparam(self):
        """Creates evals and evects. 
        
        Given a potential, builds the hamiltonian and computes its eigen basis 
        and eigen values. 
        
        Compute a vector with the eigenvalues(eV) and another with the 
        eigenvectors ((Aº)**-1/2) of the quantum hamiltonian with potential 
        (self.potential [eV]) (each column is an eigenvector with [N]+1 
        components). 
        
           H · phi = E · phi  ;  H = -(1/2m)(d**2/dx**2) + poten(x)
                    m := mass / hbar**2  [(eV·Aº**2)**-1]
    
        
        It solves the 1D time-independent Schrödinger equation for the given 
        potential (self.potential) inside of a box [a(Aº), b(Aº)], with 
        [N] intervals. 
        """
        #Dividing the ab segment in N intervals leaves a (N+1)x(N+1) 
        #hamiltonian, where indices 0 and N correspond to the potentials 
        #barriers. The hamiltonian operator has 3 non-zero diagonals (the main 
        #diagonal, and the ones next to it), with the following elements.
        semi_diag = np.full(self.N, -1./(2.*self.m*self.deltax**2))
        main_diag = self.potential + 1./(self.m*self.deltax**2)
        #Although these walls are kept here, no change seems to happen if they 
        #are removed (setting 0 as potential barriers).
        main_diag[0] += 1000000000 #Potentials barriers
        main_diag[self.N] += 1000000000
        self.evals, self.evect = spLA.eigh_tridiagonal(main_diag, semi_diag, 
                                                          check_finite = False)
        
        #Normalization. Used trapezoids method formula and that sum(evect**2)=1
        factors =1/np.sqrt(self.deltax * \
                               (1. - np.abs(self.evect[0,:])**2/2. - \
                                               np.abs(self.evect[-1,:])**2/2.))
        #Normalized vectors (* here multiplies each factor element by each 
        #evect column)
        self.evect = self.evect * factors
  
    def comp(self):
        """Creates compo. 
        
        Given psi, creates its components in the current eigenbasis.
        """
        phipsi=np.transpose(np.transpose(np.conj(self.evect)) * self.psi)
        self.compo = self.deltax * (np.sum(phipsi, axis = 0) \
                                            - phipsi[0,:]/2. - phipsi[-1,:]/2.)
    
    def psi_init(self):
        """Creates psi and psi2.
        
        Generates the initial wave function, psi: a gaussian packet centered in
        mu0 with sigma0. Shape-wise like mesh.
        
        psi2 is the sqaure of the absolute value (very much used).
        """                          
        self.psi = np.sqrt(\
                        np.exp(-(self.mesh-self.mu0)**2/(2.*self.sigma0**2)))#\
                                #*np.exp(np.complex(0.,-1.)*self.p0*self.mesh)
        prob_psi = np.abs(self.psi)**2
        self.psi *= 1. / np.sqrt(self.deltax*\
                      (np.sum(prob_psi) - prob_psi[0]/2. - prob_psi[-1]/2.))
        self.psi2 = np.abs(self.psi)**2
       
        
    ###########################################################################
    #                             PLOTTING FUNCTIONS                          #
    ###########################################################################
#    fill_bkg:
#        Paints the color zones of the plot, the area under the potential
#        and the area under psi as well.
#    fill_zones:
#        Paints the zones plot (under main animated plot) and builds redzone.
#    measure_arrow:
#        Draws arrows indicators of the result of a measure.
#    fake_axis:
#        Draws arrows and annotations as fake axis.
    
    
    def fill_bkg(self):
        """'Paints' the main plot: potential, color zones and 'psi'.
        
        Called every frame since psi changes.
        
        There is no black fill under psi (hence ''). Simply the background is
        set to black and no fill occur under psi's curve.
        
        Two fills take place here:
            potential fill -- always over psi's curve and bellow potential line
            zones fill -- always above psi's curvre and potential line
        
        What curve is on top has to be checked explicitly.
        
        Creates zones in main plot from one central given position and width.
        """
        self.bkg_twin.collections.clear() #Not drawing on top

        
        left = self.zones_center - self.zones_width/2.
        right = self.zones_center + self.zones_width/2.
        left_index = int((left - self.a)//self.deltax)
        right_index = int((right - self.a)//self.deltax)
        
        
        #Left red
        bot = (self.psi2)[:left_index+1] #Psi line
        top = np.zeros_like(bot) + 2. #Arbitrarily
        redzone = self.mesh[:left_index+1] #+1 due to slice
        potential = self.potential[:left_index+1]/self.dcoord_factor
        self.bkg_twin.fill_between(redzone, bot, potential,
                                   where = np.less(bot, potential), 
                                   facecolor = self.potcol)
        self.bkg_twin.fill_between(redzone, np.maximum(potential,bot), top,
                                   facecolor = self.zonecol_red)
        
        
        #Green
        bot = (self.psi2)[left_index-1:right_index+2]
        top = np.zeros_like(bot) + 2.
        greenzone = self.mesh[left_index-1:right_index+2] #(")
        potential=self.potential[left_index-1:right_index+2]/self.dcoord_factor
        self.bkg_twin.fill_between(greenzone, bot, potential, 
                                   where = np.less(bot, potential), 
                                   facecolor = self.potcol) #Potential
        self.bkg_twin.fill_between(greenzone, np.maximum(potential, bot),
                                   top, facecolor = self.zonecol_green) #Green
        
        
        #Right red 
        bot = (self.psi2)[right_index:] #Psi line
        top = np.zeros_like(bot) + 2.
        redzone = self.mesh[right_index:] #+1 due to slice
        potential = self.potential[right_index:]/self.dcoord_factor
        self.bkg_twin.fill_between(redzone, bot, potential,
                                   where = np.less(bot, potential), 
                                   facecolor = self.potcol) #Potential
        self.bkg_twin.fill_between(redzone, np.maximum(potential,bot), top,
                                   facecolor = self.zonecol_red) #Red
        
        
    def fill_zones(self):
        """Fill zones in zones plot and builds redzone.
        
        Zones created using central position and width.
        
        No need to be called every frame (doesn't involves psi). Called every
        new level.
        """
        self.zones.collections.clear() #Not drawing on top
        self.redzone = np.array([])
        a = 1
        
        
        left = self.zones_center - self.zones_width/2.
        right = self.zones_center + self.zones_width/2.
        left_index = int((left - self.a)//self.deltax)
        right_index = int((right - self.a)//self.deltax)
        
        
        #First red
        redzone = self.mesh[:left_index+1] #+1 due to slice
        self.redzone = np.append(self.redzone, redzone)
        bot = np.zeros_like(redzone)
        self.zones.fill_between(redzone, bot, bot + 1, 
                                facecolor = self.zonecol_red, alpha = a)
        
        
        #Green                                                            
        greenzone = self.mesh[left_index-1:right_index+2] 
        bot = np.zeros_like(greenzone)
        self.zones.fill_between(greenzone, bot, bot + 1,
                                facecolor = self.zonecol_green, alpha = a) 
        
        
        #Second red
        redzone = self.mesh[right_index:] #+1 due to slice
        self.redzone = np.append(self.redzone, redzone)
        bot = np.zeros_like(redzone)
        self.zones.fill_between(redzone, bot, bot + 1, 
                                facecolor = self.zonecol_red, alpha = a)
        
    def measure_arrow(self):
        """Draws arrows indicators of the result of a measure.
        
        Called after every measure. Fades away (this happens in plotpsiev).
        
        Draws the annotation (line) on the measured mu0. Two annotations: line
        from bottom to the instant probability, and another from there to the 
        max probability.
        """
        self.b_arrow.remove() #Not drawing on top.
        self.u_arrow.remove()
    
    
        prob = self.psi2 #%·Å^-1
        m_prob = prob[int((self.mu0 - self.a)/self.deltax)]
        max_prob = np.max(prob)
        
        
        self.b_arrow = self.bkg_twin.arrow(self.mu0, 0, 0, m_prob,
                                           color = self.b_arrow_color, 
                                           width = 0.25, head_width = 0.001, 
                                           head_length = 0.001)
        self.u_arrow = self.bkg_twin.arrow(self.mu0, m_prob, 
                                           0, max_prob - m_prob,
                                           color = self.u_arrow_color, 
                                           width = 0.25, head_width = 0.001, 
                                           head_length = 0.001)
        
        
    def fake_axis(self):
        """ Draws arrow and annotations as fake axis.
        
        This has to be made in the sake of faster frame rate. Drawing the axis
        are what takes up all the plotting time.
        """
        #POT AXIS
        fakeax_x = self.b + 0.5
        fakeax_text = 0.5
        midh = (self.pot_tlim - self.pot_blim)/2.
        fweight = 600
        self.pot_twin.annotate('', xy = (fakeax_x, self.pot_tlim),       #Arrow
                               xytext = (fakeax_x, self.pot_blim),
                               annotation_clip = False,
                               arrowprops = dict(arrowstyle = '->', 
                                                 color = self.potcol))
        self.pot_twin.text(fakeax_x + fakeax_text, midh,             #POTENTIAL 
                           'Potential [' +  self.unit_energy +']',
                           weight = fweight, color = self.potcol, 
                           rotation = 90, va = 'center', ha = 'center')
        self.pot_twin.text(fakeax_x + fakeax_text, self.pot_blim,      #Bot lim
                           str(self.pot_blim), 
                           weight = fweight, color = self.potcol,
                           va = 'center', ha = 'center')
        self.pot_twin.text(fakeax_x + fakeax_text, self.pot_tlim,      #Top lim
                           str(self.pot_tlim), 
                           weight = fweight, color = self.potcol,
                           va = 'center', ha = 'center')
        
        
        #BKG AXIS 
        midh = (self.psi_tlim - self.psi_blim)/2.
        ax_col = 'gray'
        self.bkg_twin.annotate('', xy = (-fakeax_x, self.psi_tlim),      #Arrow
                               xytext = (-fakeax_x, self.psi_blim),
                               annotation_clip = False,
                               arrowprops = dict(arrowstyle = '->', 
                                                 color = ax_col))
        self.bkg_twin.text(-fakeax_x - fakeax_text, midh,                 #PROB
                           'Probability [' +  self.unit_probxlong +']',
                           weight = fweight, color = ax_col, 
                           rotation = 90, va = 'center', ha = 'center')
        self.bkg_twin.text(-fakeax_x - fakeax_text, self.psi_blim,     #Bot lim
                           str(self.psi_blim), 
                           weight = fweight, color = ax_col,
                           va = 'center', ha = 'center')
        self.bkg_twin.text(-fakeax_x - fakeax_text, self.psi_tlim,     #Top lim
                           str(self.psi_tlim), 
                           weight = fweight, color = ax_col,
                           va = 'center', ha = 'center')
        
        
        #X AXIS
        fakeax_y = self.psi_tlim + 0.05
        fakeax_text = 0.1
        ax_col = 'white'
        fweight = 500
        self.bkg_twin.annotate('', xy = (self.b, fakeax_y),              #Arrow
                               xytext = (self.a, fakeax_y),
                               annotation_clip = False,
                               arrowprops = dict(arrowstyle = '->', 
                                                 color = ax_col))
        self.bkg_twin.text(0, fakeax_y + fakeax_text,                        #X
                           'x [' +  self.unit_long +']',
                           weight = fweight, color = ax_col, 
                           va = 'center', ha = 'center') 
        self.bkg_twin.text(self.a, fakeax_y + fakeax_text,            #Left lim 
                           str(int(self.a)), 
                           weight = fweight, color = ax_col,
                           va = 'center', ha = 'center')
        self.bkg_twin.text(self.b, fakeax_y + fakeax_text,           #Right lim
                           str(int(self.b)), 
                           weight = fweight, color = ax_col,
                           va = 'center', ha = 'center')
        
        
    ###########################################################################
    #                              GAME FUNCTIONS                             #
    ###########################################################################
#    live_resources:
#        Updates lives images when loosing/gaining them.
#    _keyboard_closed:
#        Actions taken when the keyboard is released: unbind and setting 
#        keyboard to None.
#    request_KB:
#        Bounded to the whole plot box, binds keyboard and optionally 
#        restarts the game.
#    _on_keyboard_down:
#        Functions that keyboard is actually bound to: keyboard triggers
#        measure.
#    helping_mode:
#        Switch helping state. If helping, measure results not random, 
#        always the maximum probability position.
#    transition_GS:
#        Does the transition to the starting screen, including calls to 
#        other methods apart from moving to that screen.
#    transition_GI:
#        Does the transition to the illustrative screen, including calls to 
#        other methods apart from moving to that screen.
#    generate_lvl_settings:
#        Creates the potential and the position of the green zone.
#    lvl_up:
#        Changes the speed and green zone width after leveling up.

    def lives_sources(self):
        """Updates lives images when loosing/gaining them.
        
        Called from measure, when new position is in the redzone.
        
        Replaces every live image source: having N lives, replaces live1 to
        liveN with 'heart_img.jpg', and live(N+1) to live(max_lives) with 
        'skull_img.jpg'. So, when changing the amount of lives, this has to be
        called.
        """
        for live_spot in range(1, self.max_lives+1):
            live_name = 'live' + str(live_spot)
            img = self.ids[live_name] #self.ids is a dict of all id(s) from kv
            if live_spot <= self.lives: #Heart
                img.source = self.kivy_heart_img
            else:
                img.source = self.kivy_skull_img
 
    def _keyboard_closed(self):
        """When keyboard is released unbinds it and sets it to None.
        
        Setting it to None allows to check if keyboard is bound.
        """
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None
        
    def request_KB(self, *args):
        """Binds keyboard and restarts the game if it's over.
        
        Called from a click anywhere in the plot box.
        
        Requesting and binding keyboard again, only if it has been released. 
        In game over clicking will restart the game
        """
        if self._keyboard == None and self.lives>0: 
            self._keyboard = Window.request_keyboard(self._keyboard_closed, 
                                                     self)
            self._keyboard.bind(on_key_down=self._on_keyboard_down)
        
        if self.lives <= 0: #In game over, clicking will restart.
            self._keyboard = Window.request_keyboard(self._keyboard_closed, 
                                                     self)
            self._keyboard.bind(on_key_down=self._on_keyboard_down)
            self.restart()
            
            
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        """Functions the keyboard is bound to.
        
        Bound to the event on_key_down: whenever keyboard is used this function
        will be called. So, it contains every function related to the keyboard.
        Here only to space but could be more.
        """
        if keycode[1] == 'spacebar' and self.manager.current != 'starting':
             self.measure()
        return
    
    def helping_mode(self):
        """Switch helping state.
        
        Changes the game mode to picking the most probable value for x instead
        of randomly. Changes the controlling variable and updates button label.
        """
        if self.helping: 
            self.helping = False
            self.helping_btn.text = 'OFF'
            self.helping_btn.background_color = (0, 0, 0, 1)
        else:
            self.helping = True
            self.helping_btn.text = 'ON'
            self.helping_btn.background_color = (0.133, 0.4, 0.4, 1)
        
    def transition_GS(self):
        """Screen transition: from 'gaming' to 'starting'.
        
        Includes:
            Stoping gaming screen animation loop.
            Moving to starting.
        """
        self.g_schedule_cancel()
        self.manager.transition = FadeTransition()
        self.manager.current = 'starting'

    def transition_GI(self):
        """Screen transition: from 'gaming' to 'illustrating'.
        
        Includes:
            Stoping gaming screen animation loop.
            Firing illustrating screen animation loop.
            Requesting keyboard for illustrating screen.
            Moving to illustrating.
        """
        self.g_schedule_cancel()
        iscreen = self.manager.get_screen('illustrating')
        iscreen.i_schedule_fired()
        iscreen.request_KB()
        self.manager.transition = SlideTransition()
        self.manager.transition.direction = 'left'
        self.manager.current = 'illustrating'
        
    def generate_lvl_settings(self):
        """Creates the potential and the position of the green zone.
        
        Called when passing a level.
        
        Generate random plausible levels (which consist of an harmonic
        potential and a redzone). 
        
        Some parameters will be randomly chosen from a given values interval:
            
            - harm k (arbitrary)
            - harm dx (depends on: wall_E, max_E)(biased pick)
            - redzone central position (depends on: low_E)
            
        Complementing this, redzone WIDTH will be decreasing as lvl pass and 
        SPEED will increase in the same way.
        """
            #POTENTIAL (harmonic): k, dx
        
        
        #k
        interv_k = np.linspace(3, 8, 31, endpoint = True) 
        k = np.random.choice(interv_k) #unifrom
        
        
        #dx. 
        #Its interval is such that, being the new measure mu0: 
            #1. Along all dx, the potential at the walls is greater than wall_E
            #2. The harmonic energy along all dx interval doesn't gives the 
                #position mu0 more energy than max_E        
        wall_E = 110.0 #The lower, the closer to the wall
        wall_dist = np.sqrt(2*wall_E/k)
        max_E = 90.0 #The higher, the faster it gets & the closer to wall_E 
        #(can get even higher) the more probable it hist the wall. You can see
        #its effect on how far the new wells appear from mu0.
        dx_interv_width = np.sqrt(2*max_E/k)
        left_dx = self.mu0 - dx_interv_width
        right_dx = self.mu0 + dx_interv_width
        if abs(self.a - left_dx) < wall_dist: #Building final interval
            left_dx = self.a + wall_dist
        if abs(self.b - right_dx) < wall_dist:
            right_dx = self.b - wall_dist
        inter_dx = np.arange(left_dx, right_dx, self.deltax)
        prob = (inter_dx - self.mu0)**2
        prob /= sum(prob)
        dx = np.random.choice(inter_dx, p = prob)
        self.potential = 0.5*k*(self.mesh - dx)**2
        
        
            #REDZONES
            
            
        #Central pos. 
        #It must not appear in high energy position since the particle would 
        #never reach there. So a low E value is given and the central pos 
        #interval will be on positions where low energy is enough to get there.
        low_E = 15.0 #The higher, the unluckier you're gonna get.
        c_interv_width = np.sqrt(low_E*2/k)
        inter_c = np.arange(dx - c_interv_width,dx+c_interv_width,self.deltax)
        self.zones_center = np.random.choice(inter_c)
        
        
    def lvl_up(self):
        """Changes the speed and the zones width.
        
        Called after leveling up.
        """
        #New width (wood-saxon)
        final_factor = 0.5 #Parameters to get to final factor around lvl 30
        R = 17
        a = 4
        decreas_factor=(1-final_factor)/(1+np.exp((self.lvl-R)/a))+final_factor
        self.zones_width = self.init_zones_width*decreas_factor
        #New speed (lineal)
        self.plt_vel_factor = self.init_vel + self.lvl

        
class IllustratingScreen(Screen):
    """Screen containing the demos. Explicitly stops everything when it's not 
    the current screen.
    
    Managed by MyScreenManager, here called manager.
    
    This class consists of three main blocks:
        
        - COMPUTATION of a wave function's evolution on different potentials.
        - PLOTTING of this evolution.
        - DEMOS about specific cases of this problem.
    
    All functions will be organized with this structure in mind. Nevertheless,
    the core of this class are animation and events flow, managed only
    by a few functions – they actually use the functions from the said main 
    blocks.
    
        - FLOWING functions, manage the events flow.
    
    Layout
    ------
    -BoxLayout: (containing the rest of the widgets)
        *BoxLayout: (go game, starting)
            +Go game button
            +Go starting button
        *BoxLayout: (titles, main plot, play&vel)
            +BoxLayout: (titles)
                ·Prev. button
                ·BoxLayout: (dropdown label, dropdown)
                    #Dropdown label
                    #Dropdown
                ·Next button
            +BoxLayout: (main plot)
                ·Plot box
            +BoxLayout: (play&vel)
                ·Pause button
                ·BoxLayout: (vel label, slider)
                    #Vel albel
                    #Vel slider
    
    Functions
    ---------
    (init)
        __init__:
            Initiates the screen and reads settings file.
        ipseudo_init:
            First draw, first problem solve and declaration of some variables.
    (flowing)
        i_schdeule_fired:
            Starts the looping on this screen.
        i_schedule_cancel:
            Stops the looping on this screen.
        plotpsiev:
            Animation function. Repeatedly called. Draws every frame.
        meausure:
            Picks one random position with the instant wave function, checks if
            the level is passed or not, and restarts psi.
        skip_lvl:
            Measures but always passing the level. Used via jokers.
        goto_lvl:
            Initiates the selected level.
        pause:
            Changes the paused state effectively pausing or playing the game.
    (computation)
        eigenparam:
            Given a potential, builds the hamiltonian and computes its eigen
            basis and eigen values.
        comp:
            In the eigen basis, computes the components if the current psi.
        new_potential:
            Constructs the new potential corresponding the current lvl.
        psi_init:
            Creates a gaussian wave function in a given position.
    (plotting)
        fill_bkg:
            Paints the color zones of the plot, the area under the potential
            and the area under psi as well.
        fill_zones:
            Paints the zones plot (under main animated plot) and builds 
            redzone.
        measure_arrow:
            Draws arrows indicators of the result of a measure.
        fake_axis:
            Draws arrows and annotations as fake axis.
    (demos)
        _keyboard_closed:
            Actions taken when the keyboard is released: unbind and setting 
            keyboard to None.
        request_KB:
            Bounded to the whole plot box, binds keyboard and optionally 
            restarts the game.
        _on_keyboard_down:
            Functions that keyboard is actually bound to: keyboard triggers
            measure.
        transition_IS:
            Does the transition to the starting screen, including calls to 
            other methods apart from moving to that screen.
        transition_IG:
            Does the transition to the gaming screen, including calls to 
            other methods apart from moving to that screen.
    """
    
    
    ###########################################################################
    #                            INIT FUNCTIONS                               #
    ###########################################################################
#    __init__:
#        Initiates the screen and reads settings file.
#    gpseudo_init:
#        First draw, first problem solve and declaration of some variables.
 
    
    def  __init__(self, **kwargs):
        """Initiates the screen by calling Screen's __init__ and reads 
        settings.
        
        Called from My ScreenManager __init__. When called, kivy widgets still
        don't exist, so no references to any ids can be made, hence the second
        init function: 'pseudo_init'.
        """
        super(IllustratingScreen, self).__init__(**kwargs)
        
        with open('Files/illustrating_lvl_settings.txt', 'r') as f:
            self.lvl_set = []
            self.lvl_titles = []
            for item in f:
                t = eval(item.strip())
                self.lvl_set.append(t[:-1])
                self.lvl_titles.append(t[-1])
            self.num_of_lvl = len(self.lvl_titles)
        
    def ipseudo_init(self):
        """Initiates the demos itself, by doing the first solving, first 
        drawing and setting up the demos.
        
        Sets everything ready for the loop to start calling plotpsiev. Detailed
        explanation along the code.
        
        Called from MyScreenManager init after its super __init__, when ids 
        already exists.
        """
        #======================== FIRST SOLVING ===========================                           
#        Solving this problem has two parts: finding the EIGENBASIS and 
#        eigenvalues of the given hamiltonian (with a given potential), and
#        finding the COMPONENTS of the initial psi in this basis.
        
            #SOLVING 1ST PART
                                             
        #UNITS
        self.unit_time = 'fs' 
        self.unit_energy = 'eV'
        self.unit_long = '$\AA$'
        self.unit_probxlong = '$\AA^{-1}$'
        
        #CONSTANTS
        self.hbar = 0.6582   #In these general units
        self.m_elec = 0.1316 #Its the m factor explained in eigenparam function
        self.m = self.m_elec #The name 'm' is the one used inside of eigenparam
        self.dirac_sigma = 0.6

        #DISCRETIZATION
        self.a = -10.   
        self.b = 10.        
        self.N = 800
        self.deltax = (self.b - self.a)/float(self.N)
        self.mesh = np.linspace(self.a, self.b, self.N + 1)
        
        #POTENTIAL INITAIL SETTINGS
        self.lvl = 1
        self.new_potential()
        
        #EIGENBASIS
        self.eigenparam()
       
            #SOLVING 2ND PART
        
        #PSI INITIAL SETTINGS (after every measure)
        #self.p0 = 0.    
        self.sigma0 = self.dirac_sigma
        self.psi_init()  
        
        #COMPONENTS & ENERGY
        self.comp()
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)    

        #=========================== FIRST DRAW ===============================  
#        Matplotlib plotting, its figure is going to be passed to kivy's 
#        renderer (sort of the canvas) via FigureCanvasKivyAgg. The canvas then 
#        its added as a widget to the plot box (see Layout in class docstring).
#        
#        This box is the one bound to request_KB and restarting.
#        
#        The whole plot consists of three subplots (in a vertical grid):
#            -bkg_twin:
#                Main plot. Contains color zones, under the curve fills for 
#                the potential and 'psi' (later explained) and indicator 
#                arrows.  Has a sharing x axis twin:
#                    *pot_twin: Contains potential and energy.
#            -zones:
#                Under the previous plot. Contains just the color zones.
#            -visuax:
#                Under the previous plot. Visual image of the wave function
#                by a gray map.
#        
#        For the sake of faster frame rate, no axis are going to be drawn. 
#        Instead fake ones are drawn: arrow and annotations ( fake_axis() ).
        
        
        #COLORS
        self.zonecol_red = '#AA3939'
        self.zonecol_green = '#7B9F35'
        self.potcol = '#226666'
        self.cmap_name = 'gray'
        self.energycol = '#AA8439'
        self.b_arrow_color = '#C0C0C0'
        self.u_arrow_color = '#582A72'
        
        
        #LIMITS
        self.pot_tlim = 50
        self.pot_blim = 0
        self.psi_tlim = 1.7
        self.psi_blim = 0
        Dpot = self.pot_tlim - self.pot_blim
        Dpsi = self.psi_tlim - self.psi_blim
        self.dcoord_factor = Dpot / Dpsi
        
        
        #FIGURE
        self.main_fig = plt.figure()
        self.main_fig.patch.set_facecolor('black') 
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig) #Passed to kv
        self.box1.add_widget(self.main_canvas)
        self.main_canvas.bind(on_touch_up = self.request_KB) #Rebounds keyboard
        self.gs = gridspec.GridSpec(3, 1, height_ratios=[7, 1, 1], 
                          hspace=0.1, bottom = 0.05, top = 0.90) #Subplots grid
        
        
        #BACKGROUND
        self.bkg_twin = plt.subplot(self.gs[0])
        self.bkg_twin.axis([self.a, self.b, self.psi_blim, self.psi_tlim])
        self.bkg_twin.axis('off')
        self.bkg_twin.set_facecolor('black')
            #Color zones and potential
        self.fill_bkg()
            #Arrows (first drawn transparent just to create the instance)
        self.b_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.u_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        
        
        #POTENTIAL
        self.pot_twin = self.bkg_twin.twinx()
        self.pot_twin.axis([self.a, self.b, self.pot_blim, self.pot_tlim])
        self.pot_twin.axis('off')
        self.pot_twin.set_facecolor('black')
        self.pot_data, = self.pot_twin.plot(self.mesh, self.potential, 
                                            color = self.potcol)
        self.E_data, = self.pot_twin.plot(self.mesh, 
                                         np.zeros_like(self.mesh)+self.energy, 
                                         color = self.energycol, ls = '--',
                                         lw = 2)
        
        
        #AXIS
        self.fake_axis()
        
        
        #ZONES
        self.zones = plt.subplot(self.gs[1])
        self.zones.axis([self.a, self.b, 0, 1])
        self.zones.axis('off')
        self.fill_zones()
        
        
        #VISUAL
        self.visuax = plt.subplot(self.gs[2])
        self.num_visu = len(self.mesh) #Can't be greater than the # of indices
        self.inter_visu = 'gaussian'
        self.visuax.axis('off')
        step = int(len(self.psi)/self.num_visu) #num_visu points in gray map
        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
                 aspect='auto', interpolation = self.inter_visu, 
                 cmap = self.cmap_name)
        
        
        #FINALLY, ACTUAL FIRST DRAW
        #This 'tight' needs to be at the end so it considers all objects drawn
        self.main_fig.tight_layout() #Fills all available space
        self.main_canvas.draw()
        
        #============================= OTHER ================================== 
        #KEYBOARD
        #request_keyboard returns an instance that represents the events on 
        #the keyboard. It can give two events(which can be bound to functions).
        #Furthermore, each event comes with some input arguments:
        #on_key_down    --      keycode, text, modifiers
        #on_key_up      --      keycode
        #Using the functions bind and unbind on this instance one can bind or 
        #unbind the instance event with a callback/function AND passing the
        #above arguments into the function.
        #When using request_keyboard a callback/function must be given. It is 
        #going to be called when releasing/closing the keyboard (shutting the 
        #app, or reassigning the keyboard with another request_keyboard). It 
        #usually unbinds everything bound.
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down = self._on_keyboard_down)
        
        
        #DROPDOWN
        #Which is going to be opened from the dropdownbtn created in .kv
        self.illu_dropdown = DropDown()
        self.illu_dropdown.max_height = Window.height/2
        for lvl in range(1, self.num_of_lvl + 1):
            btn = Button(id = str(lvl), text = self.lvl_titles[lvl - 1], 
                         size_hint_y = None, height=44, 
                         background_color = ( .5,.5, .5, .5),
                         font_name = '8_bit_madness')
            btn.font_size = btn.height
            btn.bind(on_press = self.goto_lvl)
            self.illu_dropdown.add_widget(btn)
   
    
        #TIME
        self.plt_time = 0.
        self.plt_dt = 1./30.
        self.plt_vel_factor = 16 #Factor in dt
        self.pause_state = True #Begins paused
          
                     
        #LABELS
        self.label_vel.text = 'VELOCITY    ' + str(self.plt_vel_factor) + 'X'
        
        
        
    ###########################################################################
    #                          FLOWING FUNCTIONS                              #
    ###########################################################################
#    i_schdeule_fired:
#        Starts the looping on this screen.
#    i_schedule_cancel:
#        Stops the looping on this screen.
#    plotpsiev:
#        Animation function. Repeatedly called. Draws every frame.
#    meausure:
#        Picks one random position with the instant wave function, checks if
#        the level is passed or not, and restarts psi.
#    skip_lvl:
#        Measures but always passing the level. Used via jokers.
#    goto_lvl:
#        Initiates the selected level.
#    pause:
#        Changes the paused state effectively pausing or unpausing the game.
    #'THE' CORE
    
    def i_schedule_fired(self):
        """Starts the looping on illustrating screen.
        
        Called when transitioning between screens.
        
        schdeule_interval orders kivy to try calling the given callback 
        repeatedly with the given dt. If can't be done in the given dt, does it
        the faster it can any way.
        
        Separated with an individual function for variable referencing reasons.
        """
        self.schedule = Clock.schedule_interval(self.plotpsiev, 1/30)
        
    def i_schedule_cancel(self):
        """Stops the looping on illustrating screen.
        
        Called when transitioning between screens.
        
        Separated with an individual function for variable referencing reasons.
        """
        self.schedule.cancel()
        
    
    def plotpsiev(self, dt):
        """Computes psi(t), update zones and fading.
        
        If playing: updates psi, bkg_twin and fading images alpha, 
        and draws. If paused: draws anyway but no updating occur.
        
        Called from loop, started by schedule_interval.
        
        Updates by doing a time step, computing the new psi (given the 
        components and the eigen values), and refilling bkg_twin and reshowing
        visuax image. Fasing images have their alpha set as well.
        
        Has to have dt as argument. Is the real time between frames.
        """

        if not self.pause_state: 
            #TIME
            self.plt_time += self.plt_vel_factor*self.plt_dt #Time step          
            t = self.plt_time
            
            
            #COMPUTE PSIEV(t). We do it with two steps (given t).
                #_1_. 
            #Column vector containing the product between component and 
            #exponential factor. 1st build array ary and then col. matrix 'mtx'
            ary_compexp = self.compo * \
                        np.exp(np.complex(0.,-1.)*self.evals*t/(50*self.hbar))
            mtx_compexp = np.matrix(np.reshape(ary_compexp, (self.N + 1,1)))
                #_2_. 
            #Psi(t)
            col_psi = self.evect * mtx_compexp #Matrix product
            self.psi = np.array(np.reshape(col_psi, self.N + 1))[0]
            self.psi2 = np.abs(self.psi)**2
        
        
        
            #ANIMATION (BKG & VISU & FADING ARROWS)
            self.fill_bkg()
            self.b_arrow.set_alpha(np.exp(-t/10))
            self.u_arrow.set_alpha(np.exp(-t/10))
            self.visu_im.remove()
            step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
            self.visu_im = self.visuax.imshow([self.psi2[::step]], 
                 aspect='auto', interpolation = self.inter_visu, 
                 cmap = self.cmap_name)
            
        #DRAW 
        #(keeps drawing even if there hasn't been an update)
        self.main_canvas.draw()
        
    def measure(self):
        """Pick a random position, check if its in redzone and initiates psi.
        
        Called from the keyboard (space).
        
        Picks new position and initiates psi again. In between, the zone
        is checked looking if the new position is in the redzone.
        
        If IN, following the settings, psi and potential are initiated.
        
        If OUT, psi initiated where the new position is and potential remains
        unchanged.
        """
        #NEW MU0
        prob = self.deltax*self.psi2 #Get instant probability
        prob /= sum(prob)
        self.mu0 = np.random.choice(self.mesh, p=prob) #Pick new random mu0
        self.measure_arrow()
        self.plt_time = 0. #Reset time 
        self.sigma0 = self.dirac_sigma #New sigma
        
        
        #CHECKING NEW POSITION
        if self.mu0 in self.redzone:
            #Doesn't follow settings conditions
            self.psi_init(apply_cond = False) 
            
        else: 
            #NEW LEVEL
            self.lvl += 1 #Read new lvl
            if self.lvl > self.num_of_lvl:
                self.lvl = 1
            self.dropdownlabel.text = self.lvl_titles[self.lvl - 1]
            self.new_potential()
            self.fill_zones()
            self.pot_data.set_data(self.mesh, self.potential)
            self.eigenparam()
            
            
            #Follows settings conditions
            self.psi_init()
            
        #REST OF PSI INITIATE
        self.comp()
        self.fill_bkg()
        #VISUAL PLOT
        self.visu_im.remove()
        step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
             aspect='auto', interpolation = self.inter_visu, 
             cmap = self.cmap_name)
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)
        self.E_data.set_data(self.mesh, self.energy)
        
        
    def skip_lvl(self, step):
        """Skips the given number of levels and pauses.
        
        Called from the 'next' and 'prev' buttons, which jump +1 and -1 levels
        respectively.
        
        Does exactly what measure does, but always passes the level (no 
        checking).
        """
        #NEW MU0
        prob = self.deltax*self.psi2 #Get instant probability
        prob /= sum(prob)
        self.mu0 = np.random.choice(self.mesh, p=prob)
        self.plt_time = 0.
        self.sigma0 = self.dirac_sigma 
        self.b_arrow.remove()
        self.u_arrow.remove()
        self.b_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.u_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.pause_state = True
        self.pause_btn.text = 'PLAY'


        #NEW LVL        
        self.lvl += step
        if self.lvl > self.num_of_lvl:
            self.lvl = 1
        elif self.lvl < 1:
            self.lvl = self.num_of_lvl
        self.dropdownlabel.text = self.lvl_titles[self.lvl - 1]
        self.new_potential()
        self.fill_zones()
        self.pot_data.set_data(self.mesh, self.potential)
        self.eigenparam()
        
        
        #NEW PSI
        self.psi_init()
        self.comp()
        self.fill_bkg()
        self.visu_im.remove()
        step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
             aspect='auto', interpolation = self.inter_visu, 
             cmap = self.cmap_name)
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)
        self.E_data.set_data(self.mesh, self.energy)
        
        
         
    def goto_lvl(self, btn):
        """Initiates the selected level.
        
        Called from the buttons inside the dropdown. When pressed, goto_lvl 
        gets an instance of that button as variable.
        
        Initiates the given level. This also means resetting the time. Starts 
        paused.
        """
        #To notify the dropdown that something has been selected and close it
        self.illu_dropdown.select('select needs an argument')


        self.plt_time = 0.
        self.pause_state = True
        self.pause_btn.text = 'PLAY'
        

        #NEW LVL
        self.lvl = eval(btn.id)
        self.dropdownlabel.text = self.lvl_titles[self.lvl - 1]
        self.new_potential()
        self.fill_zones()
        self.pot_data.set_data(self.mesh, self.potential)
        self.eigenparam()
        
        
        #NEW PSI
        self.psi_init()
        self.comp()
        self.fill_bkg()
        self.visu_im.remove()
        step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
             aspect='auto', interpolation = self.inter_visu, 
             cmap = self.cmap_name)
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)
        self.E_data.set_data(self.mesh, self.energy)     
        
        
    def pause(self):
        """Changes the pause state from true to false and viceversa.
        
        Called from the button directly.
        """  
        if self.pause_state == True:
            self.pause_state = False
            self.pause_btn.text = 'PAUSE'
        else:
            self.pause_state = True 
            self.pause_btn.text = 'PLAY'
        
    ###########################################################################
    #                            COMPUTING FUNCTIONS                          #
    ###########################################################################
#   eigenparam:
#        Given a potential, builds the hamiltonian and computes its eigen
#        basis and eigen values.
#    comp:
#        In the eigen basis, computes the components if the current psi.
#    new_potential:
#        Constructs the new potential corresponding the current lvl.
#    psi_init:
#        Creates a gaussian wave function in a given position.
 
    def eigenparam(self):
        """Creates evals and evects. 
        
        Given a potential, builds the hamiltonian and computes its eigen basis 
        and eigen values. 
        
        Compute a vector with the eigenvalues(eV) and another with the 
        eigenvectors ((Aº)**-1/2) of the quantum hamiltonian with potential 
        (self.potential [eV]) (each column is an eigenvector with [N]+1 
        components). 
        
           H · phi = E · phi  ;  H = -(1/2m)(d**2/dx**2) + poten(x)
                    m := mass / hbar**2  [(eV·Aº**2)**-1]
    
        
        It solves the 1D time-independent Schrödinger equation for the given 
        potential (self.potential) inside of a box [a(Aº), b(Aº)], with 
        [N] intervals. 
        """
        #Dividing the ab segment in N intervals leaves with a (N+1)x(N+1) 
        #hamiltonian, where indices 0 and N correspond to the potentials 
        #barriers. The hamiltonian operator has 3 non-zero diagonals (the main 
        #diagonal, and the ones next to it), with the following elements.
        semi_diag = np.full(self.N, -1./(2.*self.m*self.deltax**2))
        main_diag = self.potential + 1./(self.m*self.deltax**2)
        #Although these walls are kept here, no change seems to happen if they 
        #are removed (setting 0 as potential barriers).
        main_diag[0] += 1000000000 #Potentials barriers
        main_diag[self.N] += 1000000000
        self.evals, self.evect = spLA.eigh_tridiagonal(main_diag, semi_diag, 
                                                          check_finite = False)
        
        #Normalization. Used trapezoids method formula and that sum(evect**2)=1
        factors =1/np.sqrt(self.deltax * \
                               (1. - np.abs(self.evect[0,:])**2/2. - \
                                               np.abs(self.evect[-1,:])**2/2.))
        #Normalized vectors (* here multiplies each factor element by each 
        #evect column)
        self.evect = self.evect * factors
  
    def comp(self):
        """Creates compo. 
        
        Given psi, creates its components in the current eigenbasis.
        """
        phipsi=np.transpose(np.transpose(np.conj(self.evect)) * self.psi)
        self.compo = self.deltax * (np.sum(phipsi, axis = 0) \
                                            - phipsi[0,:]/2. - phipsi[-1,:]/2.)
            
    def new_potential(self):
        """Creates the potential.
        
        Takes the actual lvl and constructs the potential from the lvl_set (it
        comes from the reading of the file when initiated this class).
        
        The potential is build with the following indications. First element 
        chooses potential:
            
            - If 0: HARMONIC
            Line: 0, mu0, dx, k, **redzone 
            
            - If 1: DOUBLE WELL (20*[HARMONIC + CG*GAUSSIAN])
            Line: 1, mu0, dx, k, mu, sigma, CG, **redzone
            
            - If 2: TRIPLE WELL (20*[HARMONIC + CG1*GAUSSIAN + CG2*GAUSSIAN2])
            Line: 2, mu0, dx, k, mu1, sigma1, CG1, mu2, sigma2, CG2, **redzone
            
            - If 3 WOOD-SAXON
            Line: 3, mu0, H, R, a, **redzone
            
            - If 4: DOUBLE WOOD-SAXON
            Line: 4, mu0, H1, R1, a1, H2, R2, a2, **redzone
            
        Where mu0 is the position where to start the new psi. If no position 
        wants to be specified, then mu0 = 100 (checked in psi_init).
        
        Number of arguments have to be calculated so filling background has 
        where to starting taking numbers.
        """
        actual_lvl_set = self.lvl_set[self.lvl - 1]
        #HARMONIC
        if actual_lvl_set[0] == 0:
            self.fill_start_i = 4            
            dx, k = actual_lvl_set[2:self.fill_start_i]
            self.potential = 20*0.5*k*(self.mesh - dx)**2
        #DOUBLE WELL    
        elif actual_lvl_set[0] == 1:
            self.fill_start_i = 7
            dx, k, mu, sigma, CG = actual_lvl_set[2:self.fill_start_i]       
            self.potential = 20*(\
                    0.5*k*(self.mesh - dx)**2
                    +\
                    CG/np.sqrt(2*np.pi*sigma**2)*\
                    np.exp(-(self.mesh-mu)**2/(2.*sigma**2)))
        #TRIPLE WELL 
        elif actual_lvl_set[0] == 2: 
            self.fill_start_i = 10
            dx,k,mu1,sigma1,CG1,mu2,sigma2,CG2=actual_lvl_set[2:self.fill_start_i]          
            self.potential = 20*(\
                    0.5*k*(self.mesh - dx)**2
                    +\
                    CG1/np.sqrt(2*np.pi*sigma1**2)*\
                    np.exp(-(self.mesh-mu1)**2/(2.*sigma1**2))
                    +\
                    CG2/np.sqrt(2*np.pi*sigma2**2)*\
                    np.exp(-(self.mesh-mu2)**2/(2.*sigma2**2)))
        #WOOD-SAXON
        elif actual_lvl_set[0] == 3: 
            self.fill_start_i = 5
            H, R, a = actual_lvl_set[2:self.fill_start_i]           
            self.potential = -H/(1+np.exp((abs(self.mesh)-R)/a)) + H
        #DOUBLE WOOD-SAXON    
        elif actual_lvl_set[0] == 4: 
            self.fill_start_i = 8
            H1, R1, a1, H2, R2, a2 = actual_lvl_set[2:self.fill_start_i]            
            WS1 = - H1/(1 + np.exp((abs(self.mesh)-R1)/a1)) + H1
            WS2 = H2/(1 + np.exp((abs(self.mesh)-R2)/a2))           
            self.potential = WS1 + WS2
            
        else:
            print('ERROR: Bad code word for potential (1st element in line).')
        
    
    def psi_init(self, apply_cond = True):
        """Creates psi and psi2.
        
        Generates the initial wave function, psi: a gaussian packet centered in
        mu0 with sigma0. Shape-wise like mesh.
        
        If apply_cond is True the new mu0 is given by the settings and not the 
        result of the measure.
        
        psi2 is the sqaure of the absolute value (very much used).
        """                 
        if apply_cond:                      
            self.mu0 = self.lvl_set[self.lvl - 1][1]
            
        self.psi = np.sqrt(\
                        np.exp(-(self.mesh-self.mu0)**2/(2.*self.sigma0**2)))#\
                                #*np.exp(np.complex(0.,-1.)*self.p0*self.mesh)
        prob_psi = np.abs(self.psi)**2
        self.psi *= 1. / np.sqrt(self.deltax*\
                      (np.sum(prob_psi) - prob_psi[0]/2. - prob_psi[-1]/2.))
        self.psi2 = np.abs(self.psi)**2
       
    ###########################################################################
    #                             PLOTTING FUNCTIONS                          #
    ###########################################################################
#    fill_bkg:
#        Paints the color zones of the plot, the area under the potential
#        and the area under psi as well.
#    fill_zones:
#        Paints the zones plot (under main animated plot) and builds 
#        redzone.
#    measure_arrow:
#        Draws arrows indicators of the result of a measure.
#    fake_axis:
#        Draws arrows and annotations as fake axis.
    
    def fill_bkg(self):
        """'Paints' the main plot: potential, color zones and 'psi'.
        
        Called every frame since psi changes.
        
        There is no black fill under psi (hence ''). Simply the background is
        set to black and no fill occur under psi's curve.
        
        Two fills take place here:
            potential fill -- always over psi's curve and bellow potential line
            zones fill -- always above psi's curve and potential line
        
        What curve is on top has to be checked explicitly.
        
        Creates zones in main plot from one central given position and width.
        """
        self.bkg_twin.collections.clear() #Not drawing on top
        
        
        prev = self.a
        actual_lvl_set = self.lvl_set[self.lvl - 1]
            
        for i in range(self.fill_start_i, len(actual_lvl_set)-1, 2):
            #Index
            prev_index = int((prev - self.a)//self.deltax)
            index = int((actual_lvl_set[i]-self.a)//self.deltax)
            nxt_index = int((actual_lvl_set[-1]- self.a)//self.deltax)
        
            #Red
            bot = (self.psi2)[prev_index:index+1] #Psi line
            top = np.zeros_like(bot) + 2.
            redzone = self.mesh[prev_index:index+1] #+1 due to slice
            potential = self.potential[prev_index:index+1]/self.dcoord_factor
            self.redzone = np.append(self.redzone, redzone)
            self.bkg_twin.fill_between(redzone, bot, potential,
                                       where = np.less(bot, potential), 
                                       facecolor = self.potcol) #Potential
            self.bkg_twin.fill_between(redzone, np.maximum(potential,bot), top,
                                       facecolor = self.zonecol_red) #Red

            #Green
            bot = (self.psi2)[index-1:nxt_index+2]
            top = np.zeros_like(bot) + 2.
            greenzone = self.mesh[index-1:nxt_index+2] #(")
            potential = self.potential[index-1:nxt_index+2]/self.dcoord_factor
            self.bkg_twin.fill_between(greenzone, bot, potential, 
                                       where = np.less(bot, potential), 
                                       facecolor = self.potcol) #Potential
            self.bkg_twin.fill_between(greenzone, np.maximum(potential, bot),
                                       top, facecolor = self.zonecol_green)
                                                                        #Green
            #Looping by giving the new prev position
            prev = self.mesh[int((actual_lvl_set[i+1]-self.a)//self.deltax)]  
            
        #Last zone red
        bot = (self.psi2)[nxt_index:]
        top = np.zeros_like(bot) + 2.
        redzone = self.mesh[nxt_index:]
        potential = self.potential[nxt_index:]/self.dcoord_factor
        self.redzone = np.append(self.redzone, redzone)
        self.bkg_twin.fill_between(redzone, bot, potential, 
                                   where = np.less(bot, potential), 
                                   facecolor = self.potcol) #Potential
        self.bkg_twin.fill_between(redzone,  np.maximum(potential,bot), top,
                                   facecolor = self.zonecol_red) #Red
        
    def fill_zones(self):
        """Fill zones in zones plot and builds redzone.
        
        Zones created using central position and width.
        
        No need to be called every frame (doesn't involves psi). Called every
        new level.
        """
        self.zones.collections.clear() #Not drawing on top.
        self.redzone = np.array([])
        
        
        prev = self.a
        a = 1
        actual_lvl_set = self.lvl_set[self.lvl - 1]
    
    
        for i in range(self.fill_start_i, len(actual_lvl_set)-1, 2):
            #Index
            prev_index = int((prev - self.a)//self.deltax)
            index = int((actual_lvl_set[i]-self.a)//self.deltax)
            nxt_index = int((actual_lvl_set[-1]- self.a)//self.deltax)
            #Red
            redzone = self.mesh[prev_index:index+1] #+1 due to slice
            self.redzone = np.append(self.redzone, redzone)
            bot = np.zeros_like(redzone)
            self.zones.fill_between(redzone, bot, bot + 1, 
                                    facecolor = self.zonecol_red, alpha = a)
            #Green                                                            
            greenzone = self.mesh[index-1:nxt_index+2] 
            bot = np.zeros_like(greenzone)
            self.zones.fill_between(greenzone, bot, bot + 1,
                                    facecolor = self.zonecol_green, alpha = a) 
            #Looping by giving the new prev position
            prev = self.mesh[int((actual_lvl_set[i+1]-self.a)//self.deltax)] 
            
            
        #Last zone red
        redzone = self.mesh[nxt_index:]
        bot = np.zeros_like(redzone)
        self.redzone = np.append(self.redzone, redzone)
        self.zones.fill_between(redzone, bot, bot + 1,
                                facecolor = self.zonecol_red, alpha = a)
        
    def measure_arrow(self):
        """Draws arrows indicators of the result of a measure.
        
        Called after every measure. Fades away (this happens in plotpsiev).
        
        Draws the annotation (line) on the measured mu0. Two annotations: line
        from bottom to the instant probilibity, and another from there to the 
        max probability.
        """
        self.b_arrow.remove()
        self.u_arrow.remove()
    
    
        prob = self.psi2 #%·Å^-1
        m_prob = prob[int((self.mu0 - self.a)/self.deltax)]
        max_prob = np.max(prob)
    
    
        self.b_arrow = self.bkg_twin.arrow(self.mu0, 0, 0, m_prob,
                                           color = self.b_arrow_color, 
                                           width = 0.25, head_width = 0.001, 
                                           head_length = 0.001)
        
        self.u_arrow = self.bkg_twin.arrow(self.mu0, m_prob, 
                                           0, max_prob - m_prob,
                                           color = self.u_arrow_color, 
                                           width = 0.25, head_width = 0.001, 
                                           head_length = 0.001)
        
    def fake_axis(self):
        """ Draws arrow and annotations as fake axis.
        
        This has to be made in the sake of faster frame rate. Drawing the axis
        are what takes up all the plotting time.
        """
        #POT AXIS
        fakeax_x = self.b + 0.5
        fakeax_text = 0.5
        midh = (self.pot_tlim - self.pot_blim)/2.
        fweight = 600
        self.pot_twin.annotate('', xy = (fakeax_x, self.pot_tlim),  #Arrow
                               xytext = (fakeax_x, self.pot_blim),
                               annotation_clip = False,
                               arrowprops = dict(arrowstyle = '->', 
                                                 color = self.potcol))
        self.pot_twin.text(fakeax_x + fakeax_text, midh,            #Potential 
                           'Potential [' +  self.unit_energy +']',
                           weight = fweight, color = self.potcol, 
                           rotation = 90, va = 'center', ha = 'center')
        self.pot_twin.text(fakeax_x + fakeax_text, self.pot_blim,   #Bot lim
                           str(self.pot_blim), 
                           weight = fweight, color = self.potcol,
                           va = 'center', ha = 'center')
        self.pot_twin.text(fakeax_x + fakeax_text, self.pot_tlim,   #Top lim
                           str(self.pot_tlim), 
                           weight = fweight, color = self.potcol,
                           va = 'center', ha = 'center')
        #BKG AXIS 
        midh = (self.psi_tlim - self.psi_blim)/2.
        ax_col = 'gray'
        self.bkg_twin.annotate('', xy = (-fakeax_x, self.psi_tlim), #Arrow
                               xytext = (-fakeax_x, self.psi_blim),
                               annotation_clip = False,
                               arrowprops = dict(arrowstyle = '->', 
                                                 color = ax_col))
        self.bkg_twin.text(-fakeax_x - fakeax_text, midh,           #Prob
                           'Probability [' +  self.unit_probxlong +']',
                           weight = fweight, color = ax_col, 
                           rotation = 90, va = 'center', ha = 'center')
        self.bkg_twin.text(-fakeax_x - fakeax_text, self.psi_blim,  #Bot lim
                           str(self.psi_blim), 
                           weight = fweight, color = ax_col,
                           va = 'center', ha = 'center')
        self.bkg_twin.text(-fakeax_x - fakeax_text, self.psi_tlim,  #Top lim
                           str(self.psi_tlim), 
                           weight = fweight, color = ax_col,
                           va = 'center', ha = 'center')
        #X AXIS
        fakeax_y = self.psi_tlim + 0.05
        fakeax_text = 0.1
        ax_col = 'white'
        fweight = 500
        self.bkg_twin.annotate('', xy = (self.b, fakeax_y),         #Arrow
                               xytext = (self.a, fakeax_y),
                               annotation_clip = False,
                               arrowprops = dict(arrowstyle = '->', 
                                                 color = ax_col))
        self.bkg_twin.text(0, fakeax_y + fakeax_text,               #X
                           'x [' +  self.unit_long +']',
                           weight = fweight, color = ax_col, 
                           va = 'center', ha = 'center') 
        self.bkg_twin.text(self.a, fakeax_y + fakeax_text,          #Left lim 
                           str(int(self.a)), 
                           weight = fweight, color = ax_col,
                           va = 'center', ha = 'center')
        self.bkg_twin.text(self.b, fakeax_y + fakeax_text,          #Right lim
                           str(int(self.b)), 
                           weight = fweight, color = ax_col,
                           va = 'center', ha = 'center')
        
    ###########################################################################
    #                             DEMOS FUNCTIONS                             #
    ###########################################################################
#   _keyboard_closed:
#        Actions taken when the keyboard is released: unbind and setting 
#        keyboard to None.
#    request_KB:
#        Bounded to the whole plot box, binds keyboard and optionally 
#        restarts the game.
#    _on_keyboard_down:
#        Functions that keyboard is actually bound to: keyboard triggers
#        measure.
#    transition_IS:
#        Does the transition to the starting screen, including calls to 
#        other methods apart from moving to that screen.
#    transition_IG:
#        Does the transition to the gaming screen, including calls to 
#        other methods apart from moving to that screen.

    def _keyboard_closed(self):
        """When keyboard is released unbinds it and sets it to None.
        
        Setting it to None allows to check if keyboard is bound.
        """
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None
        
    def request_KB(self, *args):
        """Binds keyboard.
        
        Called from a click anywhere in the plot box.
        
        Requesting and binding keyboard again, only if it has been released.
        """
        if self._keyboard == None: #It has been released
            self._keyboard = Window.request_keyboard(self._keyboard_closed, 
                                                     self)
            self._keyboard.bind(on_key_down=self._on_keyboard_down)
        else:
            pass
            
        
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        """Functions the keyboard is bound to.
        
        Bound to the event on_key_down: whenever keyboard is used this function
        will be called. So, it contains every function related to the keyboard.
        Here only to space but could be more.
        """
        if keycode[1] == 'spacebar' and self.manager.current != 'starting':
             self.measure()
        return
            
    
    def transition_IS(self):
        """Screen transition: from 'illustrating' to 'starting'.
        
        Includes:
            Stoping illustrating screen animation loop.
            Moving to starting.
        """
        self.i_schedule_cancel()
        self.manager.transition = FadeTransition()
        self.manager.current = 'starting'

    def transition_IG(self):
        """Screen transition: from 'illustrating' to 'gaming'.
        
        Includes:
            Stoping illustrating screen animation loop.
            Firing gaming screen animation loop.
            Requesting keyboard for gaming screen.
            Moving to gaming.
        """
        self.i_schedule_cancel()
        gscreen = self.manager.get_screen('gaming')
        gscreen.g_schedule_fired()
        gscreen.request_KB()
        self.manager.transition = SlideTransition()
        self.manager.transition.direction = 'right'
        self.manager.current = 'gaming'
        
        
###############################################################################
#                                APP RUN                                      #
###############################################################################

if __name__ == "__main__":
#    Window.fullscreen = 'auto' #Opens the app directly with fullscreen
    QuanticMeasuresApp().run()