#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:13:52 2019

@author: Manu Canals
"""
###############################################################################
#                               IMPORTS                                       #
###############################################################################
#KIVY'S IMPORTS
from kivy.app import App #The executable class to run de app inherits from App 
#from kivy.uix.boxlayout import BoxLayout #The main class need to inherit from 
                #it in order to have acess to the differnt boxes in the window
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg #Canvas
                                    #object in Kivy (a Figure should be given)
from kivy.clock import Clock #Tools to manage events in Kivy (used to animate)

from kivy.uix.screenmanager import ScreenManager, Screen 
from kivy.uix.screenmanager import FadeTransition, SlideTransition   
from  kivy.uix.popup import Popup
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.core.text import LabelBase
#OTHER GENERAL IMPORTS
import numpy as np
#from matplotlib.figure import Figure #This figure is the tipical one from 
#matplotlib and is the one we shall 'adapt' to kivy using FigureCanvasKivyAgg
import matplotlib.pyplot as plt #Testing plots
import matplotlib.image as mpimg #Load image
#import timeit as ti #Used to check run times
from scipy import linalg as spLA #Its attribute eigh_tridiagonal diagonalizes H
from matplotlib import gridspec


from kivy.core.window import Window #Window manegement
#Window.fullscreen = 'auto'

###############################################################################
#                             MAIN CLASS                                      #
###############################################################################



LabelBase.register(name = '8_bit_madness', 
                   fn_regular = 'Files/8_bit_madness-regular.ttf')
LabelBase.register(name = 'VT323', 
                   fn_regular = 'Files/VT323-regular.ttf')


class StartingScreen(Screen):
    
    def __init__ (self, **kwargs):
        super(StartingScreen, self).__init__(**kwargs)
        
        
    def spseudo_init(self):
        pass
        
    def transition_SI(self):
        """
        Screen transition: from 'starting' to 'illustrating'.
        """
        iscreen = self.manager.get_screen('illustrating')
        iscreen.i_schedule_fired()
        iscreen.request_KB()
        self.manager.transition = FadeTransition()
        self.manager.current = 'illustrating'

    def transition_SG(self):
        """
        Screen transition: from 'starting' to 'gaming'.
        """
        gscreen = self.manager.get_screen('gaming')
        gscreen.g_schedule_fired()
        gscreen.request_KB()
        self.manager.transition = FadeTransition()
        self.manager.current = 'gaming'
        
class InfoPopup(Popup):
    with open('Files/info_text.txt', 'r') as t:
            lines = t.readlines()
    
    intro_text = lines[0] + '\n' +lines[1] +'\n' + lines[2] + '\n' + lines[3]
    gamele_text = '\n' + lines[4] + '\n' + lines[5] + '\n' + lines[6] + '\n' +\
                         lines[7] + '\n' + lines[8] + '\n' + lines[9] + '\n' +\
                         lines[10] + '\n' + lines[11] + '\n' + lines[12] + \
                         '\n' + lines[13]
    
    def __init__(self):
        super(Popup, self).__init__()
        self.intro.text = self.intro_text
        self.gamele.text = self.gamele_text
        
        
class PhysicsPopup(Popup):
    with open('Files/physics_text.txt', 'r') as t:
            lines = t.readlines()
    
    text = lines[0] + '\n' +lines[1] +'\n' + lines[2] + '\n' + lines[3] + '\n'\
         + lines[4] + '\n' +lines[5] +'\n' + lines[6] + '\n' + lines[7] + '\n'\
         + lines[8]
    
    def __init__(self):
        super(Popup, self).__init__()  
        self.phypop.text = self.text
        
    def phypseudo_init(self):
        pass
    

class GamingScreen(Screen):
    """
    Main class. The one passed to the executable class. It has acces to  
    differents parts of the app layout (since it inherits from BoxLayout).
    
    This class consists of three main blocks:
        
        - COMPUTATION of a wave function's evolution on different potentials
        - PLOTTING of this evolution
        - GAME. A game is build on top of this problem.
    
    All functions will be organised with this structure in mind. Nevertheless,
    the core of this class are animation and events flow, managed only
    by a few functions – they actually use the functions from the main blocks.
    
    The animation (events flow) will simply consist of:
        
        - A repeatedly called function (called by Kivy's Clock) that computes
        and plots the evolution of the wave function: PLOTPSIEV.
        - A function called by the player that takes a measure of the position
        of the instantaneous wave function, and carries on all its 
        consequences: MEASURE & SKIP_LVL
        - A function that allows the player to start again after the game is 
        over: RESTART
        - And finally a couple of functions that allow pausing the game and 
        controlling its velocity: PAUSE & CHANGE_VEL
        
    These are the time FLOWING functions.
    """
    
    def  __init__(self, **kwargs):
        super(GamingScreen, self).__init__(**kwargs)#Runs also the superclass
                                                #BoxLayout's __init__ function

    def gpseudo_init(self):    
        """
        Before starting the calls of plotpsiev, with Clock: 
            
            - The evolution problem has to be solved 
            - The plots have to be initialized 
            - The game has to be set to the beggining
            - Clock's call
            
        So in init, all of this is done before calling Clock.
        """
        #======================== EVOLUTION PROBLEM ===========================   
        """                       
        Solving this problems has two parts: finding the EIGENBASIS and 
        eigenvalues of the given hamiltonian (with a given potential), and
        find the COMPONENTS of the intial psi in this basis.
        """
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
        
        #POTENTIAL INITIAL SETTINGS
#        self.settings = open('gaming_lvl_settings.txt','r')
#        self.read_settigns()
        self.mu0 = 1 #Generating the lvl its going to be used
        self.generate_lvl_settings()
        
        #EIGENBASIS
        self.eigenparam()
       
            #SOLVING 2ND PART
        
        #PSI INITIAL SETTINGS
        #After every measure
        #self.p0 = 0.    
        self.sigma0 = self.dirac_sigma
        #This first one (even after restarting)
        self.lvl = 1 #psi_init is going to use it
#        self.psi_init(apply_cond = True)  
        self.psi_init()
        
        #COMPONENTS & ENERGY
        self.comp()
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals) 
        self.energy_av = [self.energy]     

        #============================ PLOTTING ================================  
        """
        Includes the creation of all plotted objects (legend, axis, ...)
        but the updated data. This will happen in the plotting function. Here 
        we should assign the canvas (created with FigureCanvasKivyAgg) to the 
        'box' where it will be plotted in the app with:
        self.box/panel_id.add_widget(self.FigureCanvasKivyAggs_name)
        
        There are four plots: background (BKG), psi (PSI), potential (POT) and
        a gray map for visualization (VISU). They are arranged in a (2,1) grid: 
        first 3  plots on the top of the grid and the other bellow.
        
        Those three together share the x axis and are created in an specific 
        order so important content don't overlap. That is: bkg, psi and pot.
        
        Moreover, pot has an extra plot, energy line. And psi as well, two 
        arrows for visualization of the measure.
        """
        #COLORS
        
        self.zonecol_red = '#AA3939'
        self.zonecol_green = '#7B9F35'
        self.potcol = '#226666'
        self.potalpha = 0.5
        self.orange = '#AA6C39'
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
        self.axis_on = True
        self.main_fig = plt.figure()
        self.main_fig.patch.set_facecolor('black') 
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig) #Passed to kv
        self.box1.add_widget(self.main_canvas)
        self.main_canvas.bind(on_touch_up = self.request_KB)
#        self.gs = gridspec.GridSpec(2, 1, height_ratios=[7, 1], 
#                          hspace=0.1, bottom = 0.05, top = 0.90) #Subplots grid
        self.gs = gridspec.GridSpec(3, 1, height_ratios=[7, 1, 1], 
                          hspace=0.1, bottom = 0.05, top = 0.90) #Subplots grid
        
        
        #BACKGROUND
        #Their axes are going to be as psi's, since their default position 
        #suits us. The title is going to be used as xaxis label.
        self.bkg_twin = plt.subplot(self.gs[0])
        self.bkg_twin.axis([self.a, self.b, self.psi_blim, self.psi_tlim])
        figheight = self.main_fig.get_figheight() #In inches (100 p = 1 inch)
        self.bkg_twin.set_title('x [' +  self.unit_long +']', color = 'white',
                                pad=0.05*figheight*100, fontsize=10) #pad in p
        self.bkg_twin.set_ylabel('Probability [' +  self.unit_probxlong +']', 
                                 color = 'white')
        self.bkg_twin.tick_params(axis = 'x', labelbottom=False,labeltop=True, 
                                  bottom = False, top = True)
        self.bkg_twin.tick_params(colors='white')
        self.bkg_twin.set_facecolor('black')
        self.init_zones_width = 4
        self.zones_width = self.init_zones_width
        self.fill_bkg()
        
        
        #Arrows (first drawn transparent just to create the instance)
        self.b_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.u_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        
        
        #POTENTIAL
        self.pot_twin = self.bkg_twin.twinx()
        self.pot_twin.axis([self.a, self.b, self.pot_blim, self.pot_tlim])
        self.pot_twin.set_ylabel('Potential [' +  self.unit_energy +']', 
                                 color = 'white')
        self.pot_twin.tick_params(axis='y', colors='white')
        self.pot_twin.set_facecolor('black')
        self.pot_data, = self.pot_twin.plot(self.mesh, self.potential, 
                                            color = self.potcol)
        
        
        #Energy
        self.E_data, = self.pot_twin.plot(self.mesh, 
                                         np.zeros_like(self.mesh)+self.energy, 
                                         color = self.energycol, ls = '--',
                                         lw = 2)
        
        
        #ZONES
        self.zones = plt.subplot(self.gs[1])
        self.zones.axis([self.a, self.b, 0, 1])
        self.zones.axis('off')
        self.fill_zones()
        
        
        #VISUAL
#        self.visuax = plt.subplot(self.gs[1])
        self.visuax = plt.subplot(self.gs[2])
        self.num_visu = len(self.mesh) #Can't be greater than the # of indices
        self.inter_visu = 'gaussian'
        self.visuax.axis('off')
        step = int(len(self.psi)/self.num_visu) #num_visu points in gray map
        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
                 aspect='auto', interpolation = self.inter_visu, 
                 cmap = self.cmap_name)
        
        
        #AXIS
        self.axis_off()
        self.fake_axis()
        
        
        #FIRST DRAW
        #This 'tight' needs to be at the end so it considers all objects drawn
        self.main_fig.tight_layout() #Fills all available space
        self.main_canvas.draw()
        
        #============================== GAME ================================== 
        
        #IMAGES
        self.gameover_imgdata=mpimg.imread('Images/gameover_img.png',0) #G.OVER
        self.heart_img = 'Images/heart_img.png'                          #HEART
        self.skull_img = 'Images/skull_img.png'                          #SKULL
        self.skull_imgdata = mpimg.imread('Images/skull_img.png', 0)
        self.init_alpha = 0.7 #Initial skull alpha
        self.skull_fading = None #Skull_fading can be either None or an 
        #instance of imshow. If None: no set_alpha will occur, else, setting
        #alpha and removes will be done. 
        self.lvlup_imgdata = mpimg.imread('Images/lvl_up.png')         #LVLUP
        self.lvlup_fading = None
        
        
        #GAME VARIABLES
        self.max_lives = 10 #If changed, kv file needs to be changed as well
        self.lives = self.max_lives 
        self.lives_sources() 
        self.dummy = False
        self.init_jokers = 3
        self.jokers = self.init_jokers
#        self.skip_btn.text = '    JOKER    \n remaining: ' + str(self.jokers)
        
        #KEYBOARD
        #request_keyboard returns an instance that represents the events on 
        #the keyboard. It can give two events (witch we can bind to functions).
        #Furthermore, each event comes with some input arguments:
        #on_key_down    --      keycode, text, modifiers
        #on_key_up      --      keycode
        #Using the functions bind and unbind on this instance we can bind or 
        #unbind the instance event with a callback/function AND passing the
        #above arguments into the function.
        #When using request_keyboard a callback/function must be given. It is 
        #going to be called when releasing/closing the keyboard (shutting the 
        #app, or reasigning the keyboard with another request_keyboard). It 
        #usualy unbinds everything bound.
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down = self._on_keyboard_down)
   
    
        #TIME
        self.plt_time = 0.
        self.plt_dt = 1./30.
        self.init_vel = 8
        self.plt_vel_factor = 8 #Factor in dt
        self.pause_state = True #Begins paused
                               
        
        #LABELS
#        self.label_vel.text = 'Velocity \n    ' + \
#        str(int(self.plt_vel_factor)) +'X'
                          
        
    ###########################################################################
    #                            CLOCK FUNCTIONS                              #
    ###########################################################################
    """
    Here all animation will be happening. The plotting function definied in
    the proper section will be called several times per second. The given 
    frame rate is going to be delayed (or not) depending on how many things
    happen in one frame. 
    """
    #'THE' CORE'
    def g_schedule_fired(self):
        self.schedule = Clock.schedule_interval(self.plotpsiev, 1/30)
        
    def g_schedule_cancel(self):
        self.schedule.cancel()

    ###########################################################################
    #                            'FLOW' FUNCTIONS                             #
    ###########################################################################
    """
    - plotpsiev
    - measure
    - restart
    - skip_lvl
    - change_vel
    - pause
    """
    
    #PLOTPSIEV
    def plotpsiev(self, dt):
        """
        Function to be called in the animation loop (clock.schedule_interval),
        it has to have dt as an argument. Here first is computed psi(t), then
        the data is updated in psi_data and finally it draws on the canvas. 
        This parameter dt is not the actual time, its only the real time 
        interval between calls. The time is obtained with self.plt_time.
        """
        if not self.pause_state: #Only evolve plt_time if we are unpaused
            #TIME
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
            col_psi = self.evect * mtx_compexp #Matrix product
            
            
            #UPDATE DATA. 
            #Since col_psi is a column vector (N+1,1) and we 
            #need to pass a list, we reshape and make it an array.
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
            self.b_arrow.set_alpha(0.5*np.exp(-t/10))
            self.u_arrow.set_alpha(0.5*np.exp(-t/10))
            curr_alpha = self.init_alpha*np.exp(-t/2)
            if self.skull_fading != None:
                self.skull_fading.set_alpha(curr_alpha)
            if self.lvlup_fading != None:
                self.lvlup_fading.set_alpha(curr_alpha)
                
            
        #DRAW 
        #(keeps drawing even if there hasn't been an update)
        self.main_canvas.draw()
        
        
    def measure(self):
        """
        Triggered from the kivy file. It takes the actual psi(t), generates 
        the probability distribution and picks the new value for mu. A new 
        initial wave function is created with a small sigma witch represents
        the delta post measure. The time needs to be reset to zero after 
        measuring. Finally, calls comp() and now plotpsiev() has everything it
        needs to continue plotting.
        
        Schedule:
            - Get instant probability
            - Pick new mu0
            - Reset time
            - New sigma 
            - Check zone:
                OUT
                * Substract points AND/OR change lives
                    !(extra with lives game mode)
                    ! Check if any lives left
                    ! Pauses the game
                    ! Game over image
                    ! Disables measures (buton and spacebar)
                    ! Enable restart button
                * New psi (psi_init)
                * New comp (eigenbassis still the same)
                * Draw(data) psi and fill
                * Fill bkg (WITH PSI, tho still the same redzone)
                * Redraw visuplot
                IN
                * Add points
                * New level
                * Read new level
                * New pot
                * Eigenparam
                * New psi (psi_init)
                * New comp (eigenbassis still the same)
                * Draw(data) psi and fill
                * Fill bkg (WITH PSI, with new redzone)
                * Update new redzone (while filling)
                * Redraw visuplot
            - Update labels
        """
        #NEW MU0
        prob = self.deltax*self.psi2 #Get instant probability
        prob /= sum(prob)
        self.mu0 = np.random.choice(self.mesh, p=prob) #Pick new random mu0
        if self.dummy:
            self.mu0 = self.mesh[np.argmax(prob)]
        self.measure_arrow()
        self.plt_time = 0. #Reset time 
        self.sigma0 = self.dirac_sigma #New sigma
        
        
        
        #OUT
        if self.mu0 in self.redzone: 
            #LOOSE LIFE
            passed = False
            self.lives -= 1
            self.lives_sources()
            
            
            #IMAGE POPUPS
            if self.skull_fading != None: #Previously exists an skull
                self.skull_fading.remove()
            self.skull_fading = self.pot_twin.imshow(self.skull_imgdata,
                                                 aspect = 'auto',
                                                 extent = [-2.5, 2.5, 10, 40],
                                                 alpha = self.init_alpha)
            if self.lvlup_fading != None: #If missed, remove prev lvlup
                self.lvlup_fading.remove()
                self.lvlup_fading = None
            
            
            #GAME OVER
            if self.lives <= 0: 
                if self.skull_fading != None: #When gaming over, remove skull
                    self.skull_fading.remove()
                    self.skull_fading = None
                self.pause_state = True
                self.pause_btn.text = 'PLAY'
                self.GMO_img = self.pot_twin.imshow(self.gameover_imgdata, 
                                  aspect = 'auto', extent = [-7.5, 7.5, 0, 40])
#                self.measure_btn.disabled = True
                self.pause_btn.disabled = True
#                self.restart_btn.disabled = False
                self._keyboard.release()
                print('E av: ', sum(self.energy_av)/len(self.energy_av), 
                      'Lvl: ', self.lvl)
                self.joker1.disabled = True
                self.joker2.disabled = True
                self.joker3.disabled = True
                
                
            #NEW PSI
#            self.psi_init(apply_cond = True)
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
            
            
        #IN
        else: 
            #PASS LVL
            passed = True
            self.lvl += 1 #Read new lvl
            self.label_lvl.text = 'LEVEL ' + str(self.lvl)
            self.lvl_up() #Width and speed changes
            
    
            #NEW LVL
#            self.read_settigns()
            self.generate_lvl_settings()
            self.fill_zones() #It could go inside generat_lvl_settings()
            self.pot_data.set_data(self.mesh, self.potential)
            self.eigenparam()
            
            
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
                
            
            #NEW PSI
#            self.psi_init(apply_cond = True)
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
            
            
        if passed:
            self.energy_av.append(self.energy)
        
    def skip_lvl(self):
        """
        Skips the current level. Does exactly what measure does, but always
        passes the level.
        In the game will be used as jokers.
        """
        #NEW MU0
        prob = self.deltax*self.psi2 #Get instant probability
        prob /= sum(prob)
        self.mu0 = np.random.choice(self.mesh, p=prob) #Pick new random mu0
        if self.dummy:
            self.mu0 = self.mesh[np.argmax(prob)]
        self.measure_arrow()
#        self.b_arrow.remove()
#        self.u_arrow.remove()
#        self.b_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
#        self.u_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.plt_time = 0. #Reset time 
        self.sigma0 = self.dirac_sigma #New sigma
        
        
        #PASS LVL        
        self.lvl += 1
        self.label_lvl.text = 'LEVEL ' + str(self.lvl)
        self.lvl_up()
        
        
        #NEW LVL
#        self.read_settigns()
        self.generate_lvl_settings()
        self.fill_zones()
        self.pot_data.set_data(self.mesh, self.potential)
        self.eigenparam()

        
        #IMAGE POPUPS
        if self.skull_fading != None:
            self.skull_fading.remove()
            self.skull_fading = None
        if self.lvlup_fading != None:
            self.lvlup_fading.remove()
            self.lvlup_fading = None
        
        
        #NEW PSI
#        self.psi_init(apply_cond=True)
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
        
        
        #JOKERS
        self.jokers -= 1
#        self.skip_btn.text = '    JOKER    \n remaining: ' + str(self.jokers)
#        if self.jokers <= 0:
#            self.skip_btn.disabled = True

    #RESTART
    def restart(self):
        """
        After the game is lost, sets everything ready to start again:
            - Clear game over image
            - Lvl 1
            - Lives and its images to max_lives
            - Pauses again (in cas we unpaused it during game over)
            - Starts reading the settings file again (lvl 1)
            - New pot (init)
            - Eigenparam
            - New mu0 (initial mu)
            - New psi (sigma already dirac's)
            - New comp 
            - Fill psi
            - Bkg fill + update redzone
            - Redraw visuplot
            - Enables measures (button and spacebar)
            - Disables restart 
            - Clears arrows
        """
        self.GMO_img.remove()
        self.lvl = 1
        self.label_lvl.text = 'LEVEL ' + str(self.lvl)
        self.lives = self.max_lives
        self.lives_sources()
        self.pause_state = True
        self.pause_btn.text = 'PLAY'
#        self.settings.close() #We close and open again to start reading again
#        self.settings = open('gaming_lvl_settings.txt','r')
#        self.read_settigns()
        self.zones_width = self.init_zones_width
        self.plt_vel_factor = self.init_vel
#        self.measure_btn.disabled = False
        self.pause_btn.disabled = False
        self.request_KB()
#        self.restart_btn.disabled = True
        self.b_arrow.remove()
        self.u_arrow.remove()
        self.b_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.u_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        
        
        #NEW LVL
        self.generate_lvl_settings()
        self.fill_zones()
        self.pot_data.set_data(self.mesh, self.potential)
        self.eigenparam()


        #NEW PSI
#        self.psi_init(apply_cond=True)
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
        
        #JOKERS
        self.jokers = self.init_jokers
        self.joker1.disabled = False
        self.joker2.disabled = False
        self.joker3.disabled = False
#        self.skip_btn.disabled = False
#        self.skip_btn.text = '    JOKER    \n remaining: ' + str(self.jokers)
     
    
        
    def change_vel(self):
        """
        Changes the factor in the plot time diferential.
        """
        self.plt_vel_factor *= 2
        if self.plt_vel_factor > 32:
            self.plt_vel_factor = 1
        #Label in kivy file
#        self.label_vel.text = 'Velocity \n    ' + \
#        str(int(self.plt_vel_factor)) +'X'
        
    def pause(self):
        """
        Changes the pause state from true to false and viceversa.
        """    
        self.av_dt = 0
        self.frame_count = 0
        if self.pause_state == True: #Unpausing
            self.pause_state = False
            self.pause_btn.text = 'PAUSE'
        else:
            self.pause_state = True #Pausing
            self.pause_btn.text = 'PLAY'
        
    ###########################################################################
    #                            COMPUTING FUNCTIONS                          #
    ###########################################################################
    """
    - eigenparam
    - comp
    - read_setings
    - psi_init
    - shift_psi
    """
 
    def eigenparam(self):
        """
        Compute a vector with the eigenvalues(eV) and another with the 
        eigenvectors ((Aº)**-1/2) of the quantum hamiltonian with potential 
        [self.potential [eV]] (each column is an eigenvector with [N]+1 
        components). 
        
           H · phi = E · phi  ;  H = -(1/2m)(d**2/dx**2) + poten(x)
                    m := mass / hbar**2  [(eV·Aº**2)**-1]
    
        
        It solves the 1D time-independent Schrödinger equation for the given 
        potential (self.potential) inside of a box [a(Aº), b(Aº)], with 
        [N] intervals. 
        """
        #Dividing the ab segment in N intervals leave us with a (N+1)x(N+1) 
        #hamiltonian, where indices 0 and N correspond to the potentials 
        #barriers. The hamiltonian operator has 3 non-zero diagonals (the main 
        #diagonal, and the ones next to it), with the following elements.
        semi_diag = np.full(self.N, -1./(2.*self.m*self.deltax**2))
        main_diag = self.potential + 1./(self.m*self.deltax**2)
        #Although we keep these walls here, no change seems to happen if we 
        #remove them (if we don't assign these values)
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
        """
        Generates the initial wave function's components on the eigenbasis 
        stored in self.compo.
        """
        #Compute psi components
        phipsi=np.transpose(np.transpose(np.conj(self.evect)) * self.psi)
        self.compo = self.deltax * (np.sum(phipsi, axis = 0) \
                                            - phipsi[0,:]/2. - phipsi[-1,:]/2.)
            
#    def read_settigns(self):
#        """
#        Reads file settings, assigns parameters and initializes the potentials.
#        First element chooses potential:
#            
#            - If 0: HARMONIC
#            Line: 0, mu0, dx, k, **redzone 
#            
#            - If 1: DOUBLE WELL (20*[HARMONIC + CG*GAUSSIAN])
#            Line: 1, mu0, dx, k, mu, sigma, CG, **redzone
#            
#            - If 2: TRIPLE WELL (20*[HARMONIC + CG1*GAUSSIAN + CG2*GAUSSIAN2])
#            Line: 2, mu0, dx, k, mu1, sigma1, CG1, mu2, sigma2, CG2, **redzone
#            
#            - If 3 WOOD-SAXON
#            Line: 3, mu0, H, R, a, **redzone
#            
#            - If 4: DOUBLE WOOD-SAXON
#            Line: 4, mu0, H1, R1, a1, H2, R2, a2, **redzone
#            
#        Where mu0 is the position where to start the new psi. If no position 
#        wants to be specified, then mu0 = 100 (checked in psi_init)
#        Number of arguments have to be passed to the realted variable.
#        """
#        self.lvl_set = np.array(eval(self.settings.readline().strip()))
#        #HARMONIC
#        if self.lvl_set[0] == 0: 
#            dx, k = self.lvl_set[2:3+1]
#            self.potential = 0.5*k*(self.mesh - dx)**2
#            self.fill_start_i = 4
#         #DOUBLE WELL    
#        elif self.lvl_set[0] == 1:
#            dx, k, mu, sigma, CG = self.lvl_set[2:6+1]       
#            self.potential = 20*(\
#                    0.5*k*(self.mesh - dx)**2
#                    +\
#                    CG/np.sqrt(2*np.pi*sigma**2)*\
#                    np.exp(-(self.mesh-mu)**2/(2.*sigma**2)))
#            self.fill_start_i = 7
#        #TRIPLE WELL 
#        elif self.lvl_set[0] == 2: 
#            dx, k, mu1, sigma1, CG1, mu2, sigma2, CG2 = self.lvl_set[2:9+1]          
#            self.potential = 20*(\
#                    0.5*k*(self.mesh - dx)**2
#                    +\
#                    CG1/np.sqrt(2*np.pi*sigma1**2)*\
#                    np.exp(-(self.mesh-mu1)**2/(2.*sigma1**2))
#                    +\
#                    CG2/np.sqrt(2*np.pi*sigma2**2)*\
#                    np.exp(-(self.mesh-mu2)**2/(2.*sigma2**2)))
#            self.fill_start_i = 10
#        #WOOD-SAXON
#        elif self.lvl_set[0] == 3: 
#            H, R, a = self.lvl_set[2:4+1]           
#            self.potential = -H/(1+np.exp((abs(self.mesh)-R)/a)) + H
#            self.fill_start_i = 5
#        #DOUBLE WOOD-SAXON    
#        elif self.lvl_set[0] == 4: 
#            H1, R1, a1, H2, R2, a2 = self.lvl_set[2:7+1]            
#            WS1 = - H1/(1 + np.exp((abs(self.mesh)-R1)/a1)) + H1
#            WS2 = H2/(1 + np.exp((abs(self.mesh)-R2)/a2))           
#            self.potential = WS1 + WS2
#            self.fill_start_i = 8
#            
#        else:
#            print('ERROR: Bad code word for potential (1st element in line).')
    
    def psi_init(self):#, apply_cond = False):
        """
        Creates the initial wave function, a gaussian packet in general. The 
        output's shape is the same as mesh. If apply_cond is True, some 
        specific conditions are checked and applied. Usually, it will be True
        after starting a new level for the first time.
        """                          
#        if apply_cond:
#            #Conditions on the starting postiion
#            new_mu0 = self.lvl_set[1]
#            if new_mu0 != 100:
#                self.mu0 = new_mu0
#                print('New mu0: ', new_mu0)
#            #Other conditions
#            if self.lvl == 10:
#                #Speeding up
#                self.plt_vel_factor *= 1.5
#                self.label_vel.text = 'Velocity \n    ' + \
#                                    str(int(self.plt_vel_factor)) +'X'
                
    
        #First we generate the shape of a gaussian, no need for norm. constants
        #We then normalize using the integration over the array.
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
    """
    - fill_bkg
    - measure_arrow
    - axis_off
    - fake_axis
    """
    
    def fill_bkg3(self):
        """
        Fills background in bkg axis, bkg_twin, with red and green zones of the
        current self.lvl_set. It fills above self.psi2. Keeps track of 
        the red points in self.redzone. We take every other border from the 
        file and fill the prev. zone(red) and the following zone (green). 
        Last one added a side (red).
        """
        self.bkg_twin.collections.clear() #Clear before so we don't draw on top
        self.redzone = np.array([])
        prev = self.a
            
        for i in range(self.fill_start_i, len(self.lvl_set)-1, 2):
            #Index
            prev_index = int((prev - self.a)//self.deltax)
            index = int((self.lvl_set[i]-self.a)//self.deltax)
            nxt_index = int((self.lvl_set[-1]- self.a)//self.deltax)
        
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
            prev = self.mesh[int((self.lvl_set[i+1]-self.a)//self.deltax)]  
            
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
        
        
    def fill_bkg2(self):
        """
        Fill only pot and top part gray.
        """
        self.bkg_twin.collections.clear()
        #Pot
        potential = self.potential/self.dcoord_factor
        self.bkg_twin.fill_between(self.mesh, self.psi2, potential,
                                   where = np.less(self.psi2, potential),
                                   facecolor = self.potcol)
        #Top part
        self.bkg_twin.fill_between(self.mesh, 
                                   np.maximum(potential, self.psi2),
                                   self.psi_tlim, facecolor = 'gray')
        
    def fill_bkg(self):
        """
        Creates zones from one central given position and width.
        """
        self.bkg_twin.collections.clear() #Clear before so we don't draw on top
#        self.redzone = np.array([])
        
        
        left = self.zones_center - self.zones_width/2.
        right = self.zones_center + self.zones_width/2.
        left_index = int((left - self.a)//self.deltax)
        right_index = int((right - self.a)//self.deltax)
        
        
        #Left red
        bot = (self.psi2)[:left_index+1] #Psi line
        top = np.zeros_like(bot) + 2.
        redzone = self.mesh[:left_index+1] #+1 due to slice
        potential = self.potential[:left_index+1]/self.dcoord_factor
#        self.redzone = np.append(self.redzone, redzone)
        self.bkg_twin.fill_between(redzone, bot, potential,
                                   where = np.less(bot, potential), 
                                   facecolor = self.potcol) #Potential
        self.bkg_twin.fill_between(redzone, np.maximum(potential,bot), top,
                                   facecolor = self.zonecol_red) #Red
        
        
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
#        self.redzone = np.append(self.redzone, redzone)
        self.bkg_twin.fill_between(redzone, bot, potential,
                                   where = np.less(bot, potential), 
                                   facecolor = self.potcol) #Potential
        self.bkg_twin.fill_between(redzone, np.maximum(potential,bot), top,
                                   facecolor = self.zonecol_red) #Red
        
        
    def fill_zones2(self):
        """
        
        """
        self.zones.collections.clear() #Clear before so we don't draw on top
        self.redzone = np.array([])
        prev = self.a
        a = 1
    
        for i in range(self.fill_start_i, len(self.lvl_set)-1, 2):
            #Index
            prev_index = int((prev - self.a)//self.deltax)
            index = int((self.lvl_set[i]-self.a)//self.deltax)
            nxt_index = int((self.lvl_set[-1]- self.a)//self.deltax)
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
            prev = self.mesh[int((self.lvl_set[i+1]-self.a)//self.deltax)] 
        #Last zone red
        redzone = self.mesh[nxt_index:]
        bot = np.zeros_like(redzone)
        self.redzone = np.append(self.redzone, redzone)
        self.zones.fill_between(redzone, bot, bot + 1,
                                facecolor = self.zonecol_red, alpha = a)
        
        
    def fill_zones(self):
        """
        Fill zones from center position.
        """
        self.zones.collections.clear() #Clear before so we don't draw on top
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
        """
        Draws the annotation (line) on the measured mu0. Two annotations: line
        from bottom to the probilibity we got, and another from there to the 
        max probability.
        """
        #Clears before running
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
        
    def axis_off(self):
        """
        Turns off or on the axis.
        """
        if self.axis_on: #They are on, switching them off
            self.bkg_twin.axis('off') #Difference in dt when on or off: 0.02
            self.bkg_twin.set_title(' ')      
            self.pot_twin.axis('off') #Difference in dt when on or off: 0.01
            self.axis_on = False
        else:
            self.bkg_twin.axis('on')    
            figheight = self.main_fig.get_figheight() #In inches (100p = 1inch)
            self.bkg_twin.set_title('x [' +  self.unit_long +']', 
                                    color = 'white', pad=0.05*figheight*100, 
                                    fontsize=10) #pad in p     
            self.pot_twin.axis('on')
            self.axis_on = True
            
    def fake_axis(self):
        """
        Draws arrow and text as fake axis.
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
    """
    - lives_sources
    - _keyboard_closed
    - request_KB
    - _on_keyboard_down 
    - dummy_mode
    - transition_GS
    - transition_GI
    """

    def lives_sources(self):
        """
        Replaces every live image source: having N lives, replaces live1 to
        liveN with 'heart_img.jpg', and live(N+1) to live(max_lives) with 
        'skull_img.jpg'. So, when changing the amount of lives, this has to be
        called.
        """
        for live_spot in range(1, self.max_lives+1):
            live_name = 'live' + str(live_spot)
            img = self.ids[live_name] #self.ids is a dict of all id(s) from kv
            if live_spot <= self.lives: #Heart
                img.source = self.heart_img
            else:
                img.source = self.skull_img
 
    def _keyboard_closed(self):
        """
        Actions taken when keyboard is released/closed. Unbinding and 
        'removing' the _keyboard.
        """
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        #Is happening that clicking on the box (outside any button) relases the 
        #keyboard. This can be 'fixed' adding a button that requests the 
        #keyboard again.
        self._keyboard = None
        
    def request_KB(self, *args):
        """
        Requesting and binding keyboard again, only if it has been released. 
        In gmae over clicking will restart the game
        """
        if self._keyboard == None and self.lives>0: #It has been released and 
                                                    #its not game over
            self._keyboard = Window.request_keyboard(self._keyboard_closed, 
                                                     self)
            self._keyboard.bind(on_key_down=self._on_keyboard_down)
        
        if self.lives <= 0: #In game over, clicking will restart.
            self.restart()
            
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        """
        Bind to the event on_key_down, whenever keyboard is used this function
        will be called. So, it contains every function related to the keyboard.
        """
        #We still want the escape to close the window, so diong the following,
        #pressing twice escape will close it.
        if keycode[1] == 'spacebar' and self.manager.current != 'starting':
             self.measure()
        return
    
    def dummy_mode(self):
        """
        Changes the game mode to picking the most probable value for x instead
        of randomly. Changes the controlling variable and updates button label.
        """
        if self.dummy: # Using dummy. Change mode
            self.dummy = False
            self.dummy_btn.text = 'OFF'
            self.dummy_btn.background_color = (0, 0, 0, 1)
        else: # Not using dummy. Change mode
            self.dummy = True
            self.dummy_btn.text = 'ON'
            self.dummy_btn.background_color = (0.133, 0.4, 0.4, 1)
        
    def transition_GS(self):
        self.g_schedule_cancel()
        self.manager.transition = FadeTransition()
        self.manager.current = 'starting'

    def transition_GI(self):
        self.g_schedule_cancel()
        iscreen = self.manager.get_screen('illustrating')
        iscreen.i_schedule_fired()
        iscreen.request_KB()
        self.manager.transition = SlideTransition()
        self.manager.transition.direction = 'left'
        self.manager.current = 'illustrating'
        
    def generate_lvl_settings(self):
        """
        Generate random plausible levels. This functions is going to 
        be called when passing a level. It will generate the harmonic potential
        and the redzones.
        Some parameteres will be randomly picked from a given values interval:
            - harm k (arbitrary)
            - harm dx (depends on: wall_E, max_E)(biased pick)
            - redzone central position (depends on: low_E)
        Complementing this, redzone WIDTH will be decreasing as lvl pass and 
        SPEED will increase in the same way.
        To later use fill_zones and fill_bkg we should return an array for them
        to use.
        """
        #POTENTIAL: k, dx
        #k. Its interval is almost arbitrary
        interv_k = np.linspace(3, 8, 31, endpoint = True) 
        k = np.random.choice(interv_k) #unifrom
        #dx. Its interval is such that, being the new measure mu0: 
            #1. Along all dx, the potential at the walls is greater than wall_E
            #2. The harmonic energy along all dx interval doesnt gives the 
                #position mu0 more energy than max_E        
        wall_E = 110.0 #The lower, the closer to the wall
        wall_dist = np.sqrt(2*wall_E/k)
        max_E = 90.0 #The higher, the faster it gets & the closer to wall_E 
        #(can get even higher) the more probable it hist the wall. You can see
        #its effect on how far the new wells appear from mu0.
        s = np.sqrt(2*max_E/k)
        left_dx = self.mu0 - s
        right_dx = self.mu0 + s
        if abs(self.a - left_dx) < wall_dist: #Building final interval
            left_dx = self.a + wall_dist
        if abs(self.b - right_dx) < wall_dist:
            right_dx = self.b - wall_dist
        inter_dx = np.arange(left_dx, right_dx, self.deltax)
        prob = (inter_dx - self.mu0)**2
        prob /= sum(prob)
        dx = np.random.choice(inter_dx, p = prob)
        self.potential = 0.5*k*(self.mesh - dx)**2
        
        #REDZONES: central pos, extrem values
        #Central pos. We have to be carefull that doesnt appears in a high 
        #energy region, since the particle would never get there. So we give
        #a low E value and the central pos interval will on positions where
        #low energy is enough to get there.
        low_E = 15.0 #The higher, the unlckier you're gonna get
        s = np.sqrt(low_E*2/k)
        inter_c = np.arange(dx - s, dx + s, self.deltax)
        self.zones_center = np.random.choice(inter_c)
        
    def lvl_up(self):
        """
        
        """
        #New width
        final_factor = 0.5 #Parameters to get to final factor around lvl 30
        R = 17
        a = 4
        decreas_factor=(1-final_factor)/(1+np.exp((self.lvl-R)/a))+final_factor
        self.zones_width = self.init_zones_width*decreas_factor
        #New speed
        self.plt_vel_factor = self.init_vel + self.lvl
#        self.label_vel.text = 'Velocity \n    ' + \
#                                            str(int(self.plt_vel_factor)) +'X'

        
class IllustratingScreen(Screen):
    """
    Main class. The one passed to the executable class. It has acces to  
    differents parts of the app layout (since it inherits from BoxLayout).
    
    This class consists of three main blocks:
        
        - COMPUTATION of a wave function's evolution on different potentials
        - PLOTTING of this evolution
        - GAME. A game is build on top of this problem.
    
    All functions will be organised with this structure in mind. Nevertheless,
    the core of this class are animation and events flow, managed only
    by a few functions – they actually use the functions from the main blocks.
    
    The animation (events flow) will simply consist of:
        
        - A repeatedly called function (called by Kivy's Clock) that computes
        and plots the evolution of the wave function: PLOTPSIEV.
        - A function called by the player that takes a measure of the position
        of the instantaneous wave function, and carries on all its 
        consequences: MEASURE
        - A function that allows the player to start again after the game is 
        over: RESTART
        - And finally a couple of functions that allow pausing the game and 
        controlling its velocity: PAUSE & CHANGE_VEL
        
    These are the time FLOWING functions.
    
    Before starting the calls of plotpsiev, with Clock: 
        
        - The evolution problem has to be solved 
        - The plots have to be initialized 
        - The game has to be set to the beggining
        - Clock's call
        
    So in init, all of this is done before calling Clock.
    """
 
    def  __init__(self, **kwargs):
        super(IllustratingScreen, self).__init__(**kwargs)
        
        #Reading file
        with open('Files/illustrating_lvl_settings.txt', 'r') as f:
            
            self.lvl_set = []
            self.lvl_titles = []
            for item in f:
                t = eval(item.strip())
                self.lvl_set.append(t[:-1])
                self.lvl_titles.append(t[-1])
            self.num_of_lvl = len(self.lvl_titles)
        
    def ipseudo_init(self):
        #======================== EVOLUTION PROBLEM ===========================   
        """                       
        Solving this problems has two parts: finding the EIGENBASIS and 
        eigenvalues of the given hamiltonian (with a given potential), and
        find the COMPONENTS of the intial psi in this basis.
        """
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
#        self.settings = open('illustrating_lvl_settings.txt','r')
#        self.read_settigns()
        self.new_potential()
        
        #EIGENBASIS
        self.eigenparam()
       
            #SOLVING 2ND PART
        
        #PSI INITIAL SETTINGS
        #This first one (even after restarting) and after everymeasure.
#        self.p0 = 0.    
        self.sigma0 = self.dirac_sigma
        
        self.psi_init(apply_cond = True)  
        
        #COMPONENTS & ENERGY
        self.comp()
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)                

        #============================ PLOTTING ================================  
        """
        Includes the creation of all plotted objects (legend, axis, ...)
        but the updated data. This will happen in the plotting function. Here 
        we should assign the canvas (created with FigureCanvasKivyAgg) to the 
        'box' where it will be plotted in the app with:
        self.box/panel_id.add_widget(self.FigureCanvasKivyAggs_name)
        
        There are four plots: background (BKG), psi (PSI), potential (POT) and
        a gray map for visualization (VISU). They are arranged in a (2,1) grid: 
        first 3  plots on the top of the grid and the other bellow.
        
        Those three together share the x axis and are created in an specific 
        order so important content don't overlap. That is: bkg, psi and pot.
        
        Moreover, pot has an extra plot, energy line. And psi as well, two 
        arrows for visualization of the measure.
        """
        
        #COLORS
        self.zonecol_red = '#AA3939'
        self.zonecol_green = '#7B9F35'
        self.potcol = '#226666'
        self.potalpha = 0.5
        self.orange = '#AA6C39'
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
        self.axis_on = True
        self.main_fig = plt.figure()
        self.main_fig.patch.set_facecolor('black') 
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig) #Passed to kv
        self.box1.add_widget(self.main_canvas)
#        self.gs = gridspec.GridSpec(2, 1, height_ratios=[7, 1], 
#                          hspace=0.1, bottom = 0.05, top = 0.90) #Subplots grid
        self.main_canvas.bind(on_touch_up = self.request_KB) #Rebounds keyboard
        self.gs = gridspec.GridSpec(3, 1, height_ratios=[7, 1, 1], 
                          hspace=0.1, bottom = 0.05, top = 0.90) #Subplots grid
        
        
        #BACKGROUND
        #Their axes are going to be as psi's, since their default position 
        #suits us. The title is going to be used as xaxis label.
        self.bkg_twin = plt.subplot(self.gs[0])
        self.bkg_twin.axis([self.a, self.b, self.psi_blim, self.psi_tlim])
        figheight = self.main_fig.get_figheight() #In inches (100 p = 1 inch)
        self.bkg_twin.set_title('x [' +  self.unit_long +']', color = 'white',
                                pad=0.05*figheight*100, fontsize=10) #pad in p
        self.bkg_twin.set_ylabel('Probability [' +  self.unit_probxlong +']', 
                                 color = 'white')
        self.bkg_twin.tick_params(axis = 'x', labelbottom=False,labeltop=True, 
                                  bottom = False, top = True)
        self.bkg_twin.tick_params(colors='white')
        self.bkg_twin.set_facecolor('black')
        self.fill_bkg()
        #Arrows (first drawn transparent just to create the instance)
        self.b_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.u_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        
        
        #POTENTIAL
        self.pot_twin = self.bkg_twin.twinx()
        self.pot_twin.axis([self.a, self.b, self.pot_blim, self.pot_tlim])
        self.pot_twin.set_ylabel('Potential [' +  self.unit_energy +']', 
                                 color = 'white')
        self.pot_twin.tick_params(axis='y', colors='white')
        self.pot_twin.set_facecolor('black')
        self.pot_data, = self.pot_twin.plot(self.mesh, self.potential, 
                                            color = self.potcol)
        
        #Energy
        self.E_data, = self.pot_twin.plot(self.mesh, 
                                         np.zeros_like(self.mesh)+self.energy, 
                                         color = self.energycol, ls = '--',
                                         lw = 2)
        
        
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
        
    
        #AXIS        
        self.axis_off() #Start with axis off, eventually there will be no axis
        self.fake_axis()
        
        
        #FIRST DRAW
        #This 'tight' needs to be at the end so it considers all objects drawn
        self.main_fig.tight_layout() #Fills all available space
        self.main_canvas.draw()
        
        #============================== GAME ================================== 
        
        #IMAGES
        self.gameover_imgdata = mpimg.imread('Images/gameover_img.png', 0)
        self.heart_img = 'Images/heart_img.png'
        self.skull_img = 'Images/skull_img.png'   
        
        
        #GAME VARIABLES
#        self.max_lives = 10 #If changed, kv's file needs to be changed as well
#        self.lives = self.max_lives 
#        self.lives_sources() 
#        self.dummy = False
        
        #KEYBOARD
#       request_keyboard returns an instance that represents the events on 
#       the keyboard. It can give two events (witch we can bind to functions).
#       Furthermore, each event comes with some input arguments:
#       on_key_down    --      keycode, text, modifiers
#       on_key_up      --      keycode
#       Using the functions bind and unbind on this instance we can bind or 
#       unbind the instance event with a callback/function AND passing the
#       above arguments into the function.
#       When using request_keyboard a callback/function must be given. It is 
#       going to be called when releasing/closing the keyboard (shutting the 
#       app, or reasigning the keyboard with another request_keyboard). It 
#       usualy unbinds everything bound.
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
    #                            CLOCK FUNCTIONS                              #
    ###########################################################################    
    """
    Here all animation will be happening. The plotting function definied in
    the proper section will be called several times per second. The given 
    frame rate is going to be delayed (or not) depending on how many things
    happen in one frame. 
    """
    #'THE' CORE
    def i_schedule_fired(self):
        self.schedule = Clock.schedule_interval(self.plotpsiev, 1/30)
        
    def i_schedule_cancel(self):
        self.schedule.cancel()
        

    ###########################################################################
    #                            'FLOW' FUNCTIONS                             #
    ###########################################################################
    """
    - plotpsiev
    - measure
    - restart
    - skip_lvl
    - change_vel
    - pause
    """
    
    #PLOTPSIEV
    def plotpsiev(self, dt):
        """
        Function to be called in the animation loop (clock.schedule_interval),
        it has to have dt as an argument. Here first is computed psi(t), then
        the data is updated in psi_data and finally it draws on the canvas. 
        This parameter dt is not the actual time, its only the real time 
        interval between calls. The time is obtained with self.plt_time.
        """
        if not self.pause_state: #Only evolve plt_time if we are unpaused
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
            
            #UPDATE DATA. Since col_psi is a column vector (N+1,1) and we 
            #need to pass a list, we reshape and make it an array.
            self.psi = np.array(np.reshape(col_psi, self.N + 1))[0]
            self.psi2 = np.abs(self.psi)**2
        
            #BKG
            self.fill_bkg()
            self.b_arrow.set_alpha(np.exp(-t/10))
            self.u_arrow.set_alpha(np.exp(-t/10))
            
            #VISUAL PLOT
            self.visu_im.remove()
            step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
            self.visu_im = self.visuax.imshow([self.psi2[::step]], 
                 aspect='auto', interpolation = self.inter_visu, 
                 cmap = self.cmap_name)
            
        #DRAW 
        #(keeps drawing even if there hasn't been an update)
        self.main_canvas.draw()
        
    def measure(self):
        """
        Triggered from the kivy file. It takes the actual psi(t), generates 
        the probability distribution and picks the new value for mu. A new 
        initial wave function is created with a small sigma witch represents
        the delta post measure. The time needs to be reset to zero after 
        measuring. Finally, calls comp() and now plotpsiev() has everything it
        needs to continue plotting.
        
        Schedule:
            - Get instant probability
            - Pick new mu0
            - Reset time
            - New sigma 
            - Check zone:
                OUT
                * Substract points AND/OR change lives
                    !(extra with lives game mode)
                    ! Check if any lives left
                    ! Pauses the game
                    ! Game over image
                    ! Disables measures (buton and spacebar)
                    ! Enable restart button
                * New psi (psi_init)
                * New comp (eigenbassis still the same)
                * Draw(data) psi and fill
                * Fill bkg (WITH PSI, tho still the same redzone)
                * Redraw visuplot
                IN
                * Add points
                * New level
                * Read new level
                * New pot
                * Eigenparam
                * New psi (psi_init)
                * New comp (eigenbassis still the same)
                * Draw(data) psi and fill
                * Fill bkg (WITH PSI, with new redzone)
                * Update new redzone (while filling)
                * Redraw visuplot
            - Update labels
        """
        prob = self.deltax*self.psi2 #Get instant probability
        prob /= sum(prob)
        self.mu0 = np.random.choice(self.mesh, p=prob) #Pick new random mu0
#        if self.dummy:
#            self.mu0 = self.mesh[np.argmax(prob)]
            
        self.measure_arrow()
        self.plt_time = 0. #Reset time 
        self.sigma0 = self.dirac_sigma #New sigma
        
        if self.mu0 in self.redzone: #OUT
#            self.lives -= 1
#            self.lives_sources()
            self.psi_init() 
            self.comp()
#            if self.lives <= 0: #GAME OVER
#                self.pause_state = True
#                self.pause_btn.text = 'Play'
#                self.GMO_img = self.pot_twin.imshow(self.gameover_imgdata, 
#                                  aspect = 'auto', extent = [-7.5, 7.5, 0, 40])
#                self.measure_btn.disabled = True
#                self.pause_btn.disabled = True
#                self.restart_btn.disabled = False
#                self._keyboard.release()
            self.fill_bkg()
            #VISUAL PLOT
            self.visu_im.remove()
            step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
            self.visu_im = self.visuax.imshow([self.psi2[::step]], 
                 aspect='auto', interpolation = self.inter_visu, 
                 cmap = self.cmap_name)
            
        else: #IN
            self.lvl += 1 #Read new lvl
            if self.lvl > self.num_of_lvl:
                self.lvl = 1
            self.dropdownlabel.text = self.lvl_titles[self.lvl - 1]
#            self.read_settigns()
            self.new_potential()
            self.fill_zones()
            self.pot_data.set_data(self.mesh, self.potential)
            self.eigenparam()
            self.psi_init(apply_cond=True)
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
        """
        Skips the current level. Does exactly what measure does, but always
        passes the level. Goes the specified amount up or down in levels.
        """            
        prob = self.deltax*self.psi2 #Get instant probability
        prob /= sum(prob)
        self.mu0 = np.random.choice(self.mesh, p=prob) #Pick new random mu0
#        if self.dummy:
#            self.mu0 = self.mesh[np.argmax(prob)]
        self.pause_state = True
        self.pause_btn.text = 'PLAY'
        self.b_arrow.remove()
        self.u_arrow.remove()
        self.b_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.u_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.plt_time = 0. #Reset time 
        self.sigma0 = self.dirac_sigma #New sigma
        self.lvl += step
        if self.lvl > self.num_of_lvl:
            self.lvl = 1
        elif self.lvl < 1:
            self.lvl = self.num_of_lvl
        self.dropdownlabel.text = self.lvl_titles[self.lvl - 1]
#        self.read_settigns()
        self.new_potential()
        self.fill_zones()
        self.pot_data.set_data(self.mesh, self.potential)
        #Eigenparam
        self.eigenparam()
        self.psi_init(apply_cond=True)
        self.comp()
        self.fill_bkg()
        #VISUAL PLOT
        self.visu_im.remove()
        step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
             aspect='auto', interpolation = self.inter_visu, 
             cmap = self.cmap_name)
        #ENERGY
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)
        self.E_data.set_data(self.mesh, self.energy)

#    #RESTART
#    def restart(self):
#        """
#        After the game is lost, sets everything ready to start again:
#            - Clear game over image
#            - Lvl 1
#            - Lives and its images to max_lives
#            - Pauses again (in cas we unpaused it during game over)
#            - Starts reading the settings file again (lvl 1)
#            - New pot (init)
#            - Eigenparam
#            - New mu0 (initial mu)
#            - New psi (sigma already dirac's)
#            - New comp 
#            - Fill psi
#            - Bkg fill + update redzone
#            - Redraw visuplot
#            - Enables measures (button and spacebar)
#            - Disables restart 
#            - Clears arrows
#        """
#        self.GMO_img.remove()
#        self.lvl = 1
##        self.lives = self.max_lives
##        self.lives_sources()
#        self.pause_state = True
#        self.pause_btn.text = 'Play'
##        self.settings.close() #We close and open again to start reading again
##        self.settings = open('illustrating_lvl_settings.txt','r')
##        self.read_settigns()
#        self.new_potential()
#        self.fill_zones()
#        self.pot_data.set_data(self.mesh, self.potential)
#        self.eigenparam()
#        self.psi_init(apply_cond=True)
#        self.comp()
#        self.fill_bkg()
#        self.visu_im.remove()
#        step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
#        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
#             aspect='auto', interpolation = self.inter_visu, 
#             cmap = self.cmap_name)
#        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)
#        self.E_data.set_data(self.mesh, self.energy)
#     
#        self.request_KB()
#        self.measure_btn.disabled = False
#        self.pause_btn.disabled = False
#        self.request_KB()
#        self.restart_btn.disabled = True
#        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)
#        self.E_data.set_data(self.mesh, self.energy)
#        self.b_arrow.remove()
#        self.u_arrow.remove()
#        self.b_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
#        self.u_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        
    def change_vel(self):
        """
        Changes the factor in the plot time diferential.
        """
        self.plt_vel_factor *= 2
        if self.plt_vel_factor > 32:
            self.plt_vel_factor = 1
        #Label in kivy file
        self.label_vel.text = 'Velocity \n    ' + str(self.plt_vel_factor) +'X'
        
    def pause(self):
        """
        Changes the pause state from true to false and viceversa.
        """        
        if self.pause_state == True: #Unpause
            self.pause_state = False
            self.pause_btn.text = 'PAUSE'
        else:
            self.pause_state = True #Pause
            self.pause_btn.text = 'PLAY'
        
    ###########################################################################
    #                            COMPUTING FUNCTIONS                          #
    ###########################################################################
    """
    - eigenparam
    - comp
    - read_setings
    - psi_init
    - shift_psi
    """
 
    def eigenparam(self):
        """
        Compute a vector with the eigenvalues(eV) and another with the 
        eigenvectors ((Aº)**-1/2) of the quantum hamiltonian with potential 
        [self.potential [eV]] (each column is an eigenvector with [N]+1 
        components). 
        
           H · phi = E · phi  ;  H = -(1/2m)(d**2/dx**2) + poten(x)
                    m := mass / hbar**2  [(eV·Aº**2)**-1]
    
        
        It solves the 1D time-independent Schrödinger equation for the given 
        potential (self.potential) inside of a box [a(Aº), b(Aº)], with 
        [N] intervals. 
        """
        #Dividing the ab segment in N intervals leave us with a (N+1)x(N+1) 
        #hamiltonian, where indices 0 and N correspond to the potentials 
        #barriers. The hamiltonian operator has 3 non-zero diagonals (the main 
        #diagonal, and the ones next to it), with the following elements.
        semi_diag = np.full(self.N, -1./(2.*self.m*self.deltax**2))
        main_diag = self.potential + 1./(self.m*self.deltax**2)
        #Although we keep these walls here, no change seems to happen if we 
        #remove them (if we don't assign these values)
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
        """
        Generates the initial wave function's components on the eigenbasis 
        stored in self.compo.
        """
        #Compute psi components
        phipsi=np.transpose(np.transpose(np.conj(self.evect)) * self.psi)
        self.compo = self.deltax * (np.sum(phipsi, axis = 0) \
                                            - phipsi[0,:]/2. - phipsi[-1,:]/2.)
            
    def read_settigns(self):
        """
        Reads file settings, assigns parameters and initializes the potentials.
        First element chooses potential:
            
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
        wants to be specified, then mu0 = 100 (checked in psi_init)
        Number of arguments have to be passed to the related variable.
        """
        self.lvl_set = np.array(eval(self.settings.readline().strip()))
        #HARMONIC
        if self.lvl_set[0] == 0: 
            dx, k = self.lvl_set[2:3+1]
            self.potential = 20*0.5*k*(self.mesh - dx)**2
            self.fill_start_i = 4
         #DOUBLE WELL    
        elif self.lvl_set[0] == 1:
            dx, k, mu, sigma, CG = self.lvl_set[2:6+1]       
            self.potential = 20*(\
                    0.5*k*(self.mesh - dx)**2
                    +\
                    CG/np.sqrt(2*np.pi*sigma**2)*\
                    np.exp(-(self.mesh-mu)**2/(2.*sigma**2)))
            self.fill_start_i = 7
        #TRIPLE WELL 
        elif self.lvl_set[0] == 2: 
            dx, k, mu1, sigma1, CG1, mu2, sigma2, CG2 = self.lvl_set[2:9+1]          
            self.potential = 20*(\
                    0.5*k*(self.mesh - dx)**2
                    +\
                    CG1/np.sqrt(2*np.pi*sigma1**2)*\
                    np.exp(-(self.mesh-mu1)**2/(2.*sigma1**2))
                    +\
                    CG2/np.sqrt(2*np.pi*sigma2**2)*\
                    np.exp(-(self.mesh-mu2)**2/(2.*sigma2**2)))
            self.fill_start_i = 10
        #WOOD-SAXON
        elif self.lvl_set[0] == 3: 
            H, R, a = self.lvl_set[2:4+1]           
            self.potential = -H/(1+np.exp((abs(self.mesh)-R)/a)) + H
            self.fill_start_i = 5
        #DOUBLE WOOD-SAXON    
        elif self.lvl_set[0] == 4: 
            H1, R1, a1, H2, R2, a2 = self.lvl_set[2:7+1]            
            WS1 = - H1/(1 + np.exp((abs(self.mesh)-R1)/a1)) + H1
            WS2 = H2/(1 + np.exp((abs(self.mesh)-R2)/a2))           
            self.potential = WS1 + WS2
            self.fill_start_i = 8
            
        else:
            print('ERROR: Bad code word for potential (1st element in line).')
            
    def new_potential(self):
        """
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
        wants to be specified, then mu0 = 100 (checked in psi_init)
        Number of arguments have to be passed to the related variable.
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
        
    
    def psi_init(self, apply_cond = False):
        """
        Creates the initial wave function, a gaussian packet in general. The 
        output's shape is the same as mesh. If apply_cond is True, some 
        specific conditions are checked and applied. Usually, it will be True
        after starting a new level for the first time.
        """                          
        if apply_cond:
            #Conditions on the starting postiion
            new_mu0 = self.lvl_set[self.lvl - 1][1]
            if new_mu0 != 100:
                self.mu0 = new_mu0
            #Other conditions can be added below
    
        #First we generate the shape of a gaussian, no need for norm. constants
        #We then normalize using the integration over the array.
        self.psi = np.sqrt(\
                        np.exp(-(self.mesh-self.mu0)**2/(2.*self.sigma0**2)))#\
#                                 *np.exp(np.complex(0.,-1.)*self.p0*self.mesh)
        prob_psi = np.abs(self.psi)**2
        self.psi *= 1. / np.sqrt(self.deltax*\
                      (np.sum(prob_psi) - prob_psi[0]/2. - prob_psi[-1]/2.))
        self.psi2 = np.abs(self.psi)**2
        
    def shift_psi(self, x):
        """
        Makes psi a given eigenvector of the hamiltonian but shifted a
        certain amount x. Negative x means shift to the left and vicerversa.
        """
        if x  == 0 or x <= self.a or x >= self.b:
            self.psi = self.evect[:,2]
            self.psi2 = np.abs(self.psi)**2

            return
        
        #compute how many indices are in x:
        n = int(abs(x)/self.deltax)
        if x < 0:
            eigen = self.evect[:,0]
            app = np.append(eigen, np.zeros(n))
            self.psi = app[-(self.N+1):]
            self.psi2 = np.abs(self.psi)**2

        if x > 0:
            eigen = self.evect[:,0]
            app = np.append(np.zeros(n), eigen)
            self.psi = app[:self.N + 1]
            self.psi2 = np.abs(self.psi)**2
            
        self.psi *= 1. / np.sqrt(self.deltax*\
                      (np.sum(self.psi2) - self.psi2[0]/2. - self.psi2[-1]/2.))
        self.psi2 = np.abs(self.psi)**2
       
    ###########################################################################
    #                             PLOTTING FUNCTIONS                          #
    ###########################################################################
    """
    - fill_bkg
    - measure_arrow
    - axis_off
    - fake_axis
    """
    
    def fill_bkg(self):
        """
        Fills background in bkg axis, bkg_twin, with red and green zones of the
        current self.lvl_set. It fills above self.psi2. Keeps track of 
        the red points in self.redzone. We take every other border from the 
        file and fill the prev. zone(red) and the following zone (green). 
        Last one added a side (red).
        """
        self.bkg_twin.collections.clear() #Clear before so we don't draw on top
        self.redzone = np.array([])
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
        """
        Fill zones of the middle plot.
        """
        self.zones.collections.clear() #Clear before so we don't draw on top
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
        """
        Draws the annotation (line) on the measured mu0. Two annotations: line
        from bottom to the probilibity we got, and another from there to the 
        max probability.
        """
        #Clears before running
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
        
    def axis_off(self):
        """
        Turns off or on the axis.
        """
        if self.axis_on: #They are on, switching them off
            self.bkg_twin.axis('off') #Difference in dt when on or off: 0.02
            self.bkg_twin.set_title(' ')      
            self.pot_twin.axis('off') #Difference in dt when on or off: 0.01
            self.axis_on = False
        else:
            self.bkg_twin.axis('on')    
            figheight = self.main_fig.get_figheight() #In inches (100p = 1inch)
            self.bkg_twin.set_title('x [' +  self.unit_long +']', 
                                    color = 'white', pad=0.05*figheight*100, 
                                    fontsize=10) #pad in p     
            self.pot_twin.axis('on')
            self.axis_on = True
        
    def fake_axis(self):
        """
        Draws arrow and text as fake axis.
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
    #                              GAME FUNCTIONS                             #
    ###########################################################################
    """
    - lives_sources
    - _keyboard_closed
    - request_KB
    - _on_keyboard_down 
    - dummy_mode
    - transition_IS
    - transitino_IG
    """

#    def lives_sources(self):
#        """
#        Replaces every live image source: having N lives, replaces live1 to
#        liveN with 'heart_img.jpg', and live(N+1) to live(max_lives) with 
#        'skull_img.jpg'. So, when changing the amount of lives, this has to be
#        called.
#        """
#        for live_spot in range(1, self.max_lives+1):
#            live_name = 'live' + str(live_spot)
#            img = self.ids[live_name] #self.ids is a dict of all id(s) from kv
#            if live_spot <= self.lives: #Heart
#                img.source = self.heart_img
#            else:
#                img.source = self.skull_img
# 
    def _keyboard_closed(self):
        """
        Actions taken when keyboard is released/closed. Unbinding and 
        'removing' the _keyboard.
        """
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        #Is happening that clicking on the box (outside any button) relases the 
        #keyboard. This can be 'fixed' adding a button that requests the 
        #keyboard again.
        self._keyboard = None
        
    def request_KB(self, *args):
        """
        Requesting and binding keyboard again, only if it has been released.
        """
        if self._keyboard == None: #It has been released
            self._keyboard = Window.request_keyboard(self._keyboard_closed, 
                                                     self)
            self._keyboard.bind(on_key_down=self._on_keyboard_down)
        else:
            pass
            
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        """
        Bind to the event on_key_down, whenever keyboard is used this function
        will be called. So, it contains every function related to the keyboard.
        """
        #We still want the escape to close the window, so diong the following,
        #pressing twice escape will close it.
        if keycode[1] == 'spacebar' and self.manager.current != 'starting':
             self.measure()
        return
    
#    def dummy_mode(self):
#        """
#        Changes the game mode to picking the most probable value for x instead
#        of randomly. Changes the controlling variable and updates button label.
#        """
#        if self.dummy: # Using dummy. Change mode
#            self.dummy = False
#            self.dummy_btn.text = 'Helping:\n    Off'
#        else: # Not using dummy. Change mode
#            self.dummy = True
#            self.dummy_btn.text = 'Helping:\n    On'
            
    def transition_IS(self):
        self.i_schedule_cancel()
        self.manager.transition = FadeTransition()
        self.manager.current = 'starting'

    def transition_IG(self):
        self.i_schedule_cancel()
        gscreen = self.manager.get_screen('gaming')
        gscreen.g_schedule_fired()
        gscreen.request_KB()
        self.manager.transition = SlideTransition()
        self.manager.transition.direction = 'right'
        self.manager.current = 'gaming'
        
    def goto_lvl(self, btn):
        """
        Called from the dropdown. Initiates the given level. This also means
        resetting the time. Starts paused.
        """
        self.plt_time = 0.
        self.lvl = eval(btn.id)
        self.pause_state = True
        self.pause_btn.text = 'PLAY'
        #To notify the dropdown that something has been selected and close it
        self.illu_dropdown.select('select needs an argument')
        #Update text
        self.dropdownlabel.text = self.lvl_titles[self.lvl - 1]
        #Initiate the level
        self.new_potential()
        self.fill_zones()
        self.pot_data.set_data(self.mesh, self.potential)
        self.eigenparam()
        self.psi_init(apply_cond=True)
        self.comp()
        self.fill_bkg()
        self.visu_im.remove()
        step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
             aspect='auto', interpolation = self.inter_visu, 
             cmap = self.cmap_name)
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)
        self.E_data.set_data(self.mesh, self.energy)

class MyScreenManager(ScreenManager):
    
    def  __init__(self, **kwargs):
        super(MyScreenManager, self).__init__(**kwargs)
        self.get_screen('gaming').gpseudo_init()
        self.get_screen('illustrating').ipseudo_init()
        self.get_screen('starting').spseudo_init()
        PhysicsPopup().phypseudo_init()

class ScreensApp(App):
    
    def build(self):
        return MyScreenManager()

if __name__ == '__main__':
    ScreensApp().run()