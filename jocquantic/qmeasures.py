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

class QMeasures(BoxLayout):
    """ 
    Main class. The one passed to the executable class. It has acces to the 
    differents parts of the app layout (since it inherits from BoxLayout). 
    """
    def __init__(self, **kwargs):
        super(QMeasures, self).__init__(**kwargs) #Runs also the superclass
                                                #BoxLayout's __init__ function
                                                
        # -------------------- SOME DEFINITIONS -------------------------------
        #We shall give initial values but these varibles will be changed by 
        #the values given in the app, when implemented this procedure.
        
                                ###### UNITS ######
        #time: fs
        #energy: eV
        #position: Å
        #wave function: Å**-2
        self.hbar = 0.6582   #In these general units
        
                            ###### DISCRETIZATION ######
        self.a = -10.   
        self.b = 10.        
        self.N = 800
        self.deltax = (self.b - self.a)/float(self.N)
        self.mesh = np.linspace(self.a, self.b, self.N + 1)
        
                        ###### Particle's properties ######
        self.m_elec = 0.1316 #Its the m factor explained in eigenparam function
        self.m = self.m_elec #The name 'm' is the one used inside of eigenparam
        
                    ###### Double well potential's properties ######
        #Load file with settings data and first initial settings: first 3 are
        #potential settings and the rest related data for color zones.
        self.settings = open('lvl_settings.txt','r')
        self.lvl_set = np.array(eval(self.settings.readline().strip()))
        #Parameters         
        self.pot_mu = self.lvl_set[0]
        self.pot_sigma = self.lvl_set[1]
        self.pot_k = self.lvl_set[2]
        #Potential object
        self.potential = np.zeros(self.N + 1)
        
                    ###### Eigenvalues and eigenvectors ######  
        self.evals = np.zeros(self.N + 1) #Row (N+1) vector (due to eigh func)
        self.evect = np.zeros((self.N + 1, self.N + 1)) #(N+1)x(N+1) array
        
                            ###### Wave functions ######
        #Initial (gaussian parameters)
        self.p0 = 0.    
        self.dirac_sigma = 0.4
        self.sigma0 = self.dirac_sigma
        self.init_mu0 = 1
        self.mu0 = self.init_mu0
        #Related object
        self.psi0 = np.zeros(self.N + 1)    #Value of the initial wave function
        self.comp0 = np.zeros(self.N + 1)   #Its components. They are row vects
        #Evolved
        self.psiev = np.zeros(self.N + 1)
        self.compev = np.zeros(self.N + 1)
        
                                   ###### GAME ######
        
        self.max_lives = 10 #If changed, kv's file needs to be changed as well
        self.lives = self.max_lives #initial lives
        self.lives_sources() #'Draws' hearts and skulls corresponding to lives
        #LOAD IMAGE
        path = os.path.dirname(os.path.abspath(__file__))
        self.gameover_imgdata = mpimg.imread(path + str('/gameover_img.jpg'))
        
                                ###### TEXT LABEL ######
        #GAME
        self.lvl = 1
#        self.points = 0
        
        #PROB & ENERGY
        #(this variable comes from the kivy file)
        self.label1.text = 'New mu0:   ' + '%.1f' % self.mu0 + '\n' + \
            'Prob.:             ' + '-' + '\n' + \
            'Max. prob.:    ' + '-'
        
        
                                ###### KEYBOARD ######
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
    
        
        
      

        # ------------ FIRST RUN OF EIGENPARAM AND COMP -----------------------
        
        #INIT POT, PSI AND ITS COMP (1st doing EIGENPARAM)
        self.pot_init() 
        self.eigenparam()
        self.psi0_init() 
        self.psiev = self.psi0 #Just in case we measure before running (bug)
        self.comp()  
        print(self.evect[1])

        #ENERGY
        energy = np.sum(np.abs(self.comp0)**2 * self.evals)
        self.label2.text = '<E> =  \n' + '%.3f' % energy
        
        # ---------------------- PLOT OBJECT ----------------------------------
        """
        Includes the creation of all the plotted objects (legend, axis, ...)
        but the updated data. This will happen in the plotting function. Here 
        we should assign the canvas (created with FigureCanvasKivyAgg) to the 
        'box' where it will be plotted in the app with:
        self.box/panel_id.add_widget(self.FigureCanvasKivyAggs_name)
        """
        
        #COLORS
        self.zonecol_red = '#AA3939'
        self.zonecol_green = '#7B9F35'
        self.potcol = '#226666'
        
        #LIMITS
        self.pot_tlim = 50
        self.pot_blim = 0
        self.psi_tlim = 1.7
        self.psi_blim = 0
        
        #VISU
        self.num_visu = len(self.mesh) #Can't be greater than the # of indices
        self.inter_visu = 'gaussian'
        
        #FIGURE
        self.main_fig = plt.figure()
        self.main_fig.patch.set_facecolor('black') #All background outside plot
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig) #Passed to kv
        self.box1.add_widget(self.main_canvas)
        self.gs = gridspec.GridSpec(2, 1, height_ratios=[7, 1], 
                          hspace=0.1, bottom = 0.05, top = 0.90) #Subplots grid
        
        #BACKGROUND PLOT
        #Their axes are going to be as psi's, since their default position 
        #suits us. The title its going to be used as xaxis label.
        self.bkg_twin = plt.subplot(self.gs[0])
        self.bkg_twin.axis([self.a, self.b, self.psi_blim, self.psi_tlim])
        figheight = self.main_fig.get_figheight() #In inches (100 p = 1 inch)
        self.bkg_twin.set_title('x [$\AA$]', color = 'white',
                        pad = 0.05*figheight*100, fontsize = 10) #pad in points
        self.bkg_twin.set_ylabel('Probability [$\AA^{-1}$]', color = 'white')
        self.bkg_twin.tick_params(axis = 'x', labelbottom=False,labeltop=True, 
                                  bottom = False, top = True)
        self.bkg_twin.tick_params(colors='white')
        self.bkg_twin.set_facecolor('black')

        #FILLING
        #We take every other border from the file and fill the prev. zone(red) 
        #and the following zone (green).Last one added a side (red).
        self.fill_bkg(self.psiev)
        
        #PSI PLOT
        self.psi_twin = self.bkg_twin.twinx()
        self.psi_twin.axis([self.a, self.b, self.psi_blim, self.psi_tlim])
        self.psi_twin.axis('off') #bkg axis already taken care of.
        self.psi_data, = self.psi_twin.plot(self.mesh, np.abs(self.psi0)**2,
                                            alpha = 0.0)
        self.psi_twin.set_facecolor('black')
        self.psi_twin.fill_between(self.mesh,np.abs(self.psiev)**2,
                                       facecolor = 'black')
        
        #POTENTIAL PLOT
        self.pot_twin = self.bkg_twin.twinx()
        self.pot_twin.axis([self.a, self.b, self.pot_blim, self.pot_tlim])
        self.pot_twin.set_ylabel('Potential [eV]', color = 'white')
        self.pot_twin.tick_params(axis='y', colors='white')
        self.pot_twin.set_facecolor('black')
        self.pot_data, = self.pot_twin.plot(self.mesh, self.potential)
               
        #VISUAL PLOT
        self.visuax = plt.subplot(self.gs[1])
        self.visuax.axis('off')
        step = int(len(self.psiev)/self.num_visu) #num_visu points in gray map
        self.visu_im = self.visuax.imshow([np.abs(self.psiev[::step])**2], 
                 aspect='auto', interpolation = self.inter_visu, cmap = 'gray')
         
        #FIRST DRAW
        #This 'tight' needs to be at the end so it considers all objects drawn
        self.main_fig.tight_layout() #Fills all available space
        self.main_canvas.draw_idle() 
        
        # ------------------------ CLOCK --------------------------------------
        """
        Here all animation will be happening. The plotting function definied in
        the proper section will be called several times per second. It may
        include a pause or play control (the funciton itself here in clock)
        """
        
                                ###### Time steps ######
        #PLOT TIME & VELOCITY
        self.plt_time = 0.
        self.plt_dt = 1./30.
        self.plt_vel_factor = 18 #Factor in dt
        self.label_vel.text = 'Velocity \n    ' + str(self.plt_vel_factor) +'X'
        self.pause_state = True #Begins paused
        
                                ###### Clock ######
                                    
        Clock.schedule_interval(self.plotpsiev, 1/30.)
        
    #========================== FUNCTIONS =====================================  

    #LIVES IMAGES
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
                img.source = 'heart_img.jpg'
            else:
                img.source = 'skull_img.jpg'
        
    #POTENTIAL
    def pot_init(self):
        """
        Creates the potential array object, combining the harmonic and the 
        gaussian potential. Same shape as mesh.
        """
        #First line: factor to scale the potential. 
        #Second line and third line: gaussian potential. 
        #Last line: harmonic potential.
        dx = 0
        if self.lvl == 4:
            dx = -5
        elif self.lvl == 5:
            dx = +2.5
        self.potential = 20*(\
                    1./np.sqrt(2*np.pi*self.pot_sigma**2)*\
                    np.exp(-(self.mesh-self.pot_mu)**2/(2.*self.pot_sigma**2))\
                    +\
                    0.5*self.pot_k*(self.mesh - dx)**2)
        
#        self.pot_sigma = 0.5
#        self.pot_mu = 0
#        self.pot_k = 0.2
#        
#        self.potential = 20*(\
#                    1./np.sqrt(2*np.pi*self.pot_sigma**2)*\
#                    np.exp(-(self.mesh-self.pot_mu)**2/(2.*self.pot_sigma**2))\
#                    +\
#                    0.5*self.pot_k*self.mesh**2)
        
        
    
    #PSI0
    def psi0_init(self):
        """
        Creates the initial wave function, a gaussian packet in general. The 
        output's shape is the same as mesh.
        """                          
        #First we generate the shape of a gaussian, no need for norm. constants
        #We then normalize using the integration over the array.
        self.psi0 = np.sqrt(\
                         np.exp(-(self.mesh-self.mu0)**2/(2.*self.sigma0**2)))\
                                  *np.exp(np.complex(0.,-1.)*self.p0*self.mesh)
        prob_psi0 = np.abs(self.psi0)**2
        self.psi0 *= 1. / np.sqrt(self.deltax*\
                      (np.sum(prob_psi0) - prob_psi0[0]/2. - prob_psi0[-1]/2.))
        
#        self.psi0 = self.evect[:,0]
#        print(np.shape(self.psi0))
#        self.psi0 = np.append(np.zeros(100), self.psi0)
#        print(np.shape(self.psi0)) 
#        self.psi0 = self.psi0[:self.N+1]
#        print(np.shape(self.psi0))
        
    #KEYBOARD
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
    
    #CHANGE_VEL
    def change_vel(self):
        """
        Changes the factor in the plot time diferential.
        """
        self.plt_vel_factor += 2
        if self.plt_vel_factor > 10:
            self.plt_vel_factor = 1
        #Label in kivy file
        self.label_vel.text = 'Velocity \n    ' + str(self.plt_vel_factor) +'X'
        
    #PAUSE
    def pause(self):
        """
        Changes the pause state from true to false and viceversa.
        """
        
        self.measure_btn.unbind(on_press = self.measure)
        
        if self.pause_state == True: #Unpause
            self.pause_state = False
            self.label_pause.text = 'Pause'
        else:
            self.pause_state = True #Pause
            self.label_pause.text = 'Play'
            
    #FILL BACKGROUND
    def fill_bkg(self, curve):
        """
        Fills background in bkg axis, bkg_twin, with red and green zones of the
        current self.lvl_set. It fills above the given curve (usually psiev or
        psi0). Keeps track of the red points in self.redzone.
        We take every other border from the file and fill the prev. zone(red) 
        and the following zone (green). Last one added a side (red).
        """
        self.bkg_twin.collections.clear() #Clear before so we don't draw on top
        self.redzone = np.array([])
        prev = self.a
        for i in range(3, len(self.lvl_set)-1, 2):
            #Index
            prev_index = int((prev - self.a)//self.deltax)
            nxt_index = int((self.lvl_set[-1]- self.a)//self.deltax)
            index = int((self.lvl_set[i]-self.a)//self.deltax)
            #Red
            bot = (np.abs(curve)**2)[prev_index:index+1] 
            top = np.zeros_like(bot) + 2.
            redzone = self.mesh[prev_index:index+1] #+1 due to slice
            self.redzone = np.append(self.redzone, redzone)
            self.bkg_twin.fill_between(redzone,bot,top,
                                       facecolor = self.zonecol_red)
            #Green
            bot = (np.abs(curve)**2)[index:nxt_index+1]
            top = np.zeros_like(bot) + 2.
            greenzone = self.mesh[index:nxt_index+1] #(")
            self.bkg_twin.fill_between(greenzone, bot, top,
                                       facecolor = self.zonecol_green)
            #Looping by giving the new prev position
            prev = self.mesh[int((self.lvl_set[i+1]-self.a)//self.deltax)]  
        #Last zone red
        bot = (np.abs(curve)**2)[nxt_index:]
        top = np.zeros_like(bot) + 2.
        redzone = self.mesh[nxt_index:]
        self.redzone = np.append(self.redzone, redzone)
        self.bkg_twin.fill_between(redzone,bot,top,
                                       facecolor = self.zonecol_red)
    
    #UPDATE LABELS
    def update_labels(self):
        """
        Updates probability, energy and game labels.
        """
        prob = self.deltax*np.abs(self.psiev)**2 #Get instant probability
        #PROB LABELS
        self.label1.text = 'New mu0:   ' + '%.1f' % self.mu0 + '\n' \
                           'Prob.:          ' + '%.2f' \
            %(prob[int((self.mu0 - self.a)/self.deltax)] * 100.)\
            + '\n' + \
                           'Max. prob.: ' + '%.2f' \
            %(np.max(prob) * 100.)
        #ENERGY
        energy = np.sum(np.abs(self.comp0)**2 * self.evals)
        self.label2.text = '<E> =  \n' + '%.3f' % energy
        #GAME
        self.label_lvl.text = 'Level ' + str(self.lvl)
#        self.label_points.text = str(self.points) + ' points'
        
        
    
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
            #exponential factor. 1st build array ary and then col. matrix 'mtx'
            ary_compexp = self.comp0 * \
                        np.exp(np.complex(0.,-1.)*self.evals*t/(50*self.hbar))
            mtx_compexp = np.matrix(np.reshape(ary_compexp, (self.N + 1,1)))
            #_2_. 
            #Psi(t)
            col_psiev = self.evect * mtx_compexp #Matrix product
            
            #UPDATE DATA. 
            #Since col_psiev is a column vector (N+1,1) and we 
            #need to pass a list, we reshape and make it an array.
            self.psiev = np.array(np.reshape(col_psiev, self.N + 1))[0]
            self.psi_data.set_data(self.mesh, np.abs(self.psiev)**2)
            
            #FILLING PSI
            self.psi_twin.collections.clear()
            self.psi_twin.fill_between(self.mesh, np.abs(self.psiev)**2,
                                       facecolor = 'black')
            #FILLING BKG
            self.fill_bkg(self.psiev)
            
            #VISUAL PLOT
            self.visu_im.remove()
            step = int(len(self.psiev)/self.num_visu) #Same as in the 1st plot
            self.visu_im = self.visuax.imshow([np.abs(self.psiev[::step])**2], 
                 aspect='auto', interpolation = self.inter_visu, cmap = 'gray')
      
        #DRAW 
        #(keeps drawing even if ther hasn't been an update)
        self.main_canvas.draw()
    
    #EIGENPARAM
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
    
    #COMP   
    def comp(self, do_psi = False):
        
        """
        Generates the initial wave function's components on the eigenbasis if 
        d0_psi = False (stored on comp0). If d0_psi = True computes psiev 
        components (stored in compev).
        """
        if do_psi == True:
    		#Compute psi's components
            phipsi=np.transpose(np.transpose(np.conj(self.evect)) * self.psiev)
            self.compev = self.deltax * (np.sum(phipsi, axis = 0) \
                                            - phipsi[0,:]/2. - phipsi[-1,:]/2.)
        else:
    		#Compute psi0 components
            phipsi=np.transpose(np.transpose(np.conj(self.evect)) * self.psi0)
            self.comp0 = self.deltax * (np.sum(phipsi, axis = 0) \
                                            - phipsi[0,:]/2. - phipsi[-1,:]/2.)
            
    #MEASURE
    def measure(self):
        """
        Triggered from the kivy file. It takes the actual psiev(t), generates 
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
                * New psi0 (psi0_init)
                * New comp (eigenbassis still the same)
                * Draw(data) psi and fill
                * Fill bkg (WITH PSI0, tho still the same redzone)
                * Redraw visuplot
                IN
                * Add points
                * New level
                * Read new level
                * New pot
                * Eigenparam
                * New psi0 (psi0_init)
                * New comp (eigenbassis still the same)
                * Draw(data) psi and fill
                * Fill bkg (WITH PSI0, with new redzone)
                * Update new redzone (while filling)
                * Redraw visuplot
            - Update labels
        """
        prob = self.deltax*np.abs(self.psiev)**2 #Get instant probability
        self.mu0 = np.random.choice(self.mesh, p=prob) #Pick new random mu0
        self.plt_time = 0. #Reset time 
        self.sigma0 = self.dirac_sigma #New sigma
        
        if self.mu0 in self.redzone: #Check zone
#            self.points -= 5 #Bad, out of limits
            self.lives -= 1
            self.lives_sources()
            self.psi0_init() 
            self.comp()
            self.psi_data.set_data(self.mesh, np.abs(self.psi0)**2)
            self.psi_twin.collections.clear() 
            self.psi_twin.fill_between(self.mesh, np.abs(self.psi0)**2,
                                       facecolor = 'black')
            if self.lives <= 0:
                self.pause_state = True
                self.GMO_img = self.pot_twin.imshow(self.gameover_imgdata, 
                                  aspect = 'auto', extent = [-7.5, 7.5, 0, 40])
                self.measure_btn.disabled = True
                self.restart_btn.disabled = False
                self._keyboard.release()
            self.fill_bkg(self.psi0)
            #VISUAL PLOT
            self.visu_im.remove()
            step = int(len(self.psi0)/self.num_visu) #Same as in the 1st plot
            self.visu_im = self.visuax.imshow([np.abs(self.psi0[::step])**2], 
                 aspect='auto', interpolation = self.inter_visu, cmap = 'gray')
            
            
        else:
#            self.points += 10 #level passed
            self.lvl += 1 #Read new lvl
            self.lvl_set = np.array(eval(self.settings.readline().strip()))
            #New pot
            self.pot_mu = self.lvl_set[0]
            self.pot_sigma = self.lvl_set[1]
            self.pot_k = self.lvl_set[2]
            self.pot_init()
            self.pot_data.set_data(self.mesh, self.potential)
            #Eigenparam
            self.eigenparam()
            #New psi0
            self.psi0_init()
            self.comp()
            self.psi_data.set_data(self.mesh, np.abs(self.psi0)**2)
            self.psi_twin.collections.clear()
            self.psi_twin.fill_between(self.mesh, np.abs(self.psi0)**2,
                                       facecolor = 'black')
            self.fill_bkg(self.psi0)
            #VISUAL PLOT
            self.visu_im.remove()
            step = int(len(self.psi0)/self.num_visu) #Same as in the 1st plot
            self.visu_im = self.visuax.imshow([np.abs(self.psi0[::step])**2], 
                 aspect='auto', interpolation = self.inter_visu, cmap = 'gray')
                
        self.update_labels()

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
            - New psi0 (sigma already dirac's)
            - New comp 
            - Fill psi
            - Bkg fill + update redzone
            - Redraw visuplot
            - Enables measures (button and spacebar)
            - Disables restart 
            - 
        """
        self.GMO_img.remove()
        self.lvl = 1
        self.lives = self.max_lives
        self.lives_sources()
        self.pause_state = True
        self.settings.close() #We close and open again to start reading again
        self.settings = open('lvl_settings.txt','r')
        self.lvl_set = np.array(eval(self.settings.readline().strip()))
        self.pot_mu = self.lvl_set[0]
        self.pot_sigma = self.lvl_set[1]
        self.pot_k = self.lvl_set[2]
        print(self.lvl_set)
        self.pot_init()
        self.pot_data.set_data(self.mesh, self.potential)
        self.eigenparam()
        self.mu0 = self.init_mu0
        self.psi0_init()
        self.comp()
        self.psi_data.set_data(self.mesh, np.abs(self.psi0)**2)
        self.psi_twin.collections.clear()
        self.psi_twin.fill_between(self.mesh, np.abs(self.psi0)**2,
                                       facecolor = 'black')
        self.redzone = np.array([])
        self.fill_bkg(self.psi0)
        self.visu_im.remove()
        step = int(len(self.psi0)/self.num_visu) #Same as in the 1st plot
        self.visu_im = self.visuax.imshow([np.abs(self.psi0[::step])**2], 
             aspect='auto', interpolation = self.inter_visu, cmap = 'gray')
        self.measure_btn.disabled = False
        self.request_KB()
        self.restart_btn.disabled = True
        self.update_labels()
        
        
        
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