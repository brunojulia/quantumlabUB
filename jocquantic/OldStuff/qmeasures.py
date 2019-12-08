#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:29:25 2019
Manu Canals
"""
###############################################################################
#                               IMPORTS                                       #
###############################################################################
#KIVY'S IMPORTS
from kivy.app import App #The executable class to run de app inherits from App 
from kivy.uix.boxlayout import BoxLayout #The main class need to inherit from 
                #it in order to have acess to the differnt boxes in the window
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg #Canvas
                                    #object in Kivy (a Figure should be given)
from kivy.clock import Clock #Tools to manage events in Kivy (used to animate)

from kivy.uix.screenmanager import ScreenManager, Screen

#OTHER GENERAL IMPORTS
import numpy as np
#from matplotlib.figure import Figure #This figure is the tipical one from 
#matplotlib and is the one we shall 'adapt' to kivy using FigureCanvasKivyAgg
import matplotlib.pyplot as plt #Testing plots
import matplotlib.image as mpimg #Load image
import os #Getting paths
#import timeit as ti #Used to check run times
from scipy import linalg as spLA #Its attribute eigh_tridiagonal diagonalizes H
from matplotlib import gridspec


from kivy.core.window import Window #Window manegement
#Window.fullscreen = 'auto'

###############################################################################
#                             MAIN CLASS                                      #
###############################################################################
"""
The way the app is made (with Kivy) is the following. When the file is executed
Kivy's run() function is called from QMeasuresApp class (see the 'if' at 
the end). QMeasuresApp class inherits Kivy's app properties from App. 
This class has acces to  everything in QMeasures class as well, so every 
tool (functions, plot, animations ...) are defined there.

"""

class QMeasures(Screen):
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
    def __init__(self, **kwargs):
        super(QMeasures, self).__init__(**kwargs) #Runs also the superclass
                                                #BoxLayout's __init__ function
                                                
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
        self.settings = open('lvl_settings.txt','r')
        self.read_settigns()
        
        #EIGENBASIS
        self.eigenparam()
       
            #SOLVING 2ND PART
        
        #PSI INITIAL SETTINGS
        #After every measure
#        self.p0 = 0.    
        self.sigma0 = self.dirac_sigma
        #This first one (even after restarting)
        self.lvl = 1 #psi_init is going to use it
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
#        plt.show(block=False)
        self.main_fig = plt.figure()
        self.main_fig.patch.set_facecolor('black') 
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig) #Passed to kv
        self.box1.add_widget(self.main_canvas)
        self.gs = gridspec.GridSpec(2, 1, height_ratios=[7, 1], 
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
        
        #VISUAL
        self.visuax = plt.subplot(self.gs[1])
        self.num_visu = len(self.mesh) #Can't be greater than the # of indices
        self.inter_visu = 'gaussian'
        self.visuax.axis('off')
        step = int(len(self.psi)/self.num_visu) #num_visu points in gray map
        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
                 aspect='auto', interpolation = self.inter_visu, 
                 cmap = self.cmap_name)
        
        #FIRST DRAW
        #This 'tight' needs to be at the end so it considers all objects drawn
        self.main_fig.tight_layout() #Fills all available space
        self.main_canvas.draw()
        
        #============================== GAME ================================== 
        
        #IMAGES
        path = os.path.dirname(os.path.abspath(__file__))
        self.gameover_imgdata = mpimg.imread(path + str('/gameover_img.jpg'))
        self.heart_img = 'heart_img.jpg'
        self.skull_img = 'skull_img.jpg'        
        
        #GAME VARIABLES
        self.max_lives = 10 #If changed, kv's file needs to be changed as well
        self.lives = self.max_lives 
        self.lives_sources() 
        self.lvl = 1
        self.dummy = False
        
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
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
   
        #TIME
        self.plt_time = 0.
        self.plt_dt = 1./30.
        self.plt_vel_factor = 16 #Factor in dt
        self.pause_state = True #Begins paused
                               
        #LABELS
        self.label_vel.text = 'Velocity \n    ' + str(self.plt_vel_factor) +'X'
                          
        #============================== CLOCK ================================= 
        """
        Here all animation will be happening. The plotting function definied in
        the proper section will be called several times per second. The given 
        frame rate is going to be delayed (or not) depending on how many things
        happen in one frame. 
        """
        #'THE' CORE
        Clock.schedule_interval(self.plotpsiev, 1/30)

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
        if self.dummy:
            self.mu0 = self.mesh[np.argmax(prob)]
            
        self.measure_arrow()
        self.plt_time = 0. #Reset time 
        self.sigma0 = self.dirac_sigma #New sigma
        
        if self.mu0 in self.redzone: #OUT
            self.lives -= 1
            self.lives_sources()
            self.psi_init() 
            self.comp()
            if self.lives <= 0: #GAME OVER
                self.pause_state = True
                self.pause_btn.text = 'Play'
                self.GMO_img = self.pot_twin.imshow(self.gameover_imgdata, 
                                  aspect = 'auto', extent = [-7.5, 7.5, 0, 40])
                self.measure_btn.disabled = True
                self.pause_btn.disabled = True
                self.restart_btn.disabled = False
                self._keyboard.release()
            self.fill_bkg()
            #VISUAL PLOT
            self.visu_im.remove()
            step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
            self.visu_im = self.visuax.imshow([self.psi2[::step]], 
                 aspect='auto', interpolation = self.inter_visu, 
                 cmap = self.cmap_name)
            
        else: #IN
            self.lvl += 1 #Read new lvl
            self.label_lvl.text = 'Level ' + str(self.lvl)
            self.read_settigns()
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
            
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)
        self.E_data.set_data(self.mesh, self.energy)
        
    def skip_lvl(self):
        """
        Skips the current level. Does exactly what measure does, but always
        passes the level.
        """            
        prob = self.deltax*self.psi2 #Get instant probability
        self.mu0 = np.random.choice(self.mesh, p=prob) #Pick new random mu0
        if self.dummy:
            self.mu0 = self.mesh[np.argmax(prob)]
        self.b_arrow.remove()
        self.u_arrow.remove()
        self.b_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.u_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.plt_time = 0. #Reset time 
        self.sigma0 = self.dirac_sigma #New sigma
        self.lvl += 1
        self.label_lvl.text = 'Level ' + str(self.lvl)
        self.read_settigns()
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
        self.lives = self.max_lives
        self.lives_sources()
        self.pause_state = True
        self.pause_btn.text = 'Play'
        self.settings.close() #We close and open again to start reading again
        self.settings = open('lvl_settings.txt','r')
        self.read_settigns()
        self.pot_data.set_data(self.mesh, self.potential)
        self.eigenparam()
        self.psi_init(apply_cond=True)
        self.comp()
        self.redzone = np.array([])
        self.fill_bkg()
        self.visu_im.remove()
        step = int(len(self.psi)/self.num_visu) #Same as in the 1st plot
        self.visu_im = self.visuax.imshow([self.psi2[::step]], 
             aspect='auto', interpolation = self.inter_visu, 
             cmap = self.cmap_name)
     
        self.measure_btn.disabled = False
        self.pause_btn.disabled = False
        self.request_KB()
        self.restart_btn.disabled = True
        self.energy = np.sum(np.abs(self.compo)**2 * self.evals)
        self.E_data.set_data(self.mesh, self.energy)
        self.b_arrow.remove()
        self.u_arrow.remove()
        self.b_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        self.u_arrow = self.bkg_twin.arrow(0,0,0,0, alpha = 0)
        
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
#        self.measure_btn.unbind(on_press = self.measure)        
        if self.pause_state == True: #Unpause
            self.pause_state = False
            self.pause_btn.text = 'Pause'
        else:
            self.pause_state = True #Pause
            self.pause_btn.text = 'Play'
        
        
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
        Number of arguments have to be passed to the realted variable.
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
    
    def psi_init(self, apply_cond = False):
        """
        Creates the initial wave function, a gaussian packet in general. The 
        output's shape is the same as mesh. If apply_cond is True, some 
        specific conditions are checked and applied. Usually, it will be True
        after starting a new level for the first time.
        """                          
        if apply_cond:
            #Conditions on the starting postiion
            new_mu0 = self.lvl_set[1]
            if new_mu0 != 100:
                self.mu0 = new_mu0
                print('New mu0: ', new_mu0)
            #Other conditions
            if self.lvl == 10:
                #Starting at the middle maximum and paused
                self.pause_state = True
                self.pause_btn.text = 'Play'
            if self.lvl == 22:
                #Speeding up
                self.plt_vel_factor *= 1.5
                self.label_vel.text = 'Velocity \n    ' + \
                                    str(int(self.plt_vel_factor)) +'X'
                
    
        #First we generate the shape of a gaussian, no need for norm. constants
        #We then normalize using the integration over the array.
        self.psi = np.sqrt(\
                         np.exp(-(self.mesh-self.mu0)**2/(2.*self.sigma0**2)))#\
#                                  *np.exp(np.complex(0.,-1.)*self.p0*self.mesh)
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
        
    ###########################################################################
    #                              GAME FUNCTIONS                             #
    ###########################################################################
    """
    - lives_sources
    - _keyboard_closed
    - request_KB
    - _on_keyboard_down 
    - dummy_mode
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
        print('keyboard released')
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        #Is happening that clicking on the box (outside any button) relases the 
        #keyboard. This can be 'fixed' adding a button that requests the 
        #keyboard again.
        self._keyboard = None
        
    def request_KB(self):
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
        if keycode[1] == 'spacebar':
             self.measure()
        return
    
    def dummy_mode(self):
        """
        Changes the game mode to picking the most probable value for x instead
        of randomly. Changes the controlling variable and updates button label.
        """
        if self.dummy: # Using dummy. Change mode
            self.dummy = False
            self.dummy_btn.text = 'Helping:\n    Off'
        else: # Not using dummy. Change mode
            self.dummy = True
            self.dummy_btn.text = 'Helping:\n    On'
            
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
        
class QMeasuresApp(App):
    """
    This class is the one executed when executing the file. Includes the main 
    class with the app's content. Due to kivy's syntax, the part before App in
    this class name must be also the name of the files .py and .kv 
    """
    def build(self):
        self.title = 'Quantic Measures'
        return QMeasures()

if __name__ == '__main__':
    QMeasuresApp().run()
