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
        self.dirac_sigma = 0.3
        self.sigma0 = self.dirac_sigma
        self.mu0 = 1
        #Related object
        self.psi0 = np.zeros(self.N + 1)    #Value of the initial wave function
        self.comp0 = np.zeros(self.N + 1)   #Its components. They are row vects
        #Evolved
        self.psiev = np.zeros(self.N + 1)
        self.compev = np.zeros(self.N + 1)
        
                                ###### TEXT LABEL ######
        #Game
        self.lvl = 1
        self.points = 0
        #Updating the text in the kivy's labels (these variables come from the 
        #kivy file).
        self.label1.text = 'New mu0:   ' + '%.1f' % self.mu0 + '\n' + \
            'Prob.:             ' + '-' + '\n' + \
            'Max. prob.:    ' + '-'
        
        
                                ###### KEYBOARD ######
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
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
    
        
        
      

        # ------------ FIRST RUN OF EIGENPARAM AND COMP -----------------------
        
        #Potential and wave first creation
        self.pot_init() 
        self.psi0_init() 
        self.psiev = self.psi0 #Just in case we measure before running (bug)
        #Compute eigenparam and projects the wave function
        self.eigenparam()
        self.comp()  

        #Compute and show expected energy value
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
        self.psi_tlim = 1.5
        self.psi_blim = 0
        
        #VISU
        self.num_visu = len(self.mesh) #Cannot be greater than the number of indices
        self.cir_rad = 0.2
        self.min_alpha = 0.1
        
        
        self.main_fig = plt.figure()
        self.main_fig.patch.set_facecolor('black') #All background outside plot

#        Object passed to kv
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig)
        self.box1.add_widget(self.main_canvas)
        
        #THE GRID
        self.gs = gridspec.GridSpec(2, 1, height_ratios=[7, 1])
        
        #BACKGROUND PLOT
        self.bkg_twin = plt.subplot(self.gs[0])
        self.bkg_twin.axis([self.a, self.b, self.psi_blim, self.psi_tlim])
        figheight = self.main_fig.get_figheight() #In inches (100 p = 1 inch)
        self.bkg_twin.set_title('x [$\AA$]', color = 'white'
                                , pad = 0.05*figheight*100, fontsize = 10)
        #pad is passed in points 
        self.bkg_twin.tick_params(axis = 'x', labelbottom=False,labeltop=True, 
                                  bottom = False, top = True)
        #We are using their axis for 'x' axis and 'prob' axis for the default
        #bkg axis position suits us
        self.bkg_twin.set_ylabel('Probability [$\AA^{-1}$]', color = 'white') 
#        self.bkg_twin.set_xlabel('x [$\AA$]', color = 'white')
        self.bkg_twin.tick_params(colors='white')
                
        self.bkg_twin.set_facecolor('black')

        #Filling: we take every other border from the file and fill the prev. 
        # zone(red) and the following zone (green).Last one added a side (red).
        prev = self.a
        #To keep track which points are in the red zone
        self.redzone = np.array([])
        for i in range(3, len(self.lvl_set)-1, 2):
            #Index
            prev_index = int((prev - self.a)//self.deltax)
            index = int((self.lvl_set[i]-self.a)//self.deltax)
            nxt_index = int((self.lvl_set[i+1]- self.a)//self.deltax)
            #Red
            bot = (np.abs(self.psiev)**2)[prev_index:index+1] #+1 due to slice
            top = np.zeros_like(bot) + 2.
            redzone = self.mesh[prev_index:index+1] #+1 due to slice
            self.redzone = np.append(self.redzone, redzone)
            self.bkg_twin.fill_between(redzone,bot,top,
                                       facecolor = self.zonecol_red)
            #Green
            bot = (np.abs(self.psiev)**2)[index:nxt_index+1] #(")
            top = np.zeros_like(bot) + 2.
            greenzone = self.mesh[index:nxt_index+1] #(")
            self.bkg_twin.fill_between(greenzone, bot, top,
                                       facecolor = self.zonecol_green)
            #Looping by giving the new prev position
            prev = self.mesh[int((self.lvl_set[i+1]-self.a)//self.deltax)]
        #Last zone (red)
        bot = (np.abs(self.psiev)**2)[nxt_index:]
        top = np.zeros_like(bot) + 2.
        redzone = self.mesh[nxt_index:]
        self.redzone = np.append(self.redzone, redzone)
        self.bkg_twin.fill_between(redzone,bot,top,
                                       facecolor = self.zonecol_red)
        
        
        
         #PSI PLOT
        self.psi_twin = self.bkg_twin.twinx()
#        
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
               
         
        
          #VIsualplot
        self.visuax = plt.subplot(self.gs[1])
        self.visuax.axis('off')
        step = int(len(self.psiev)/self.num_visu)
        self.visu_im = self.visuax.imshow([np.abs(self.psiev[::step])**2], 
                      aspect='auto', interpolation = 'sinc', cmap = 'gray')
        
        
#        maxpsi = np.max(np.abs(self.psiev)**2)
#        for i in range(0,len(self.psi0), int(len(self.psi0)/self.num_visu)):
#            prob_psi = np.abs(self.psiev[i])**2
#            a = prob_psi/maxpsi
#            if a > self.min_alpha :
#                self.visuax.add_patch(plt.Circle((self.mesh[i], 0), 
#                                         self.cir_rad, color='white', alpha=a))
#        
        
        
        
        
        
        
        
        
        
#        
#        #VIsualplot
#        self.visuax = plt.subplot(self.gs[1])
#        
#        self.visuax.axis([self.a, self.b, -1.0, 1.0])
#        self.visuax.axis('off')
#        
#        
#        maxpsi = np.max(np.abs(self.psiev)**2)
#        for i in range(0,len(self.psi0), int(len(self.psi0)/self.num_visu)):
#            prob_psi = np.abs(self.psiev[i])**2
#            a = prob_psi/maxpsi
#            if a > self.min_alpha :
#                self.visuax.add_patch(plt.Circle((self.mesh[i], 0), 
#                                         self.cir_rad, color='white', alpha=a))
#                
          #FIRST DRAW
        self.main_fig.tight_layout() #Fills all available space (title space).
        #This 'tight' needs to be at the end so it considers all objects drawn
        self.main_canvas.draw_idle() 
        
        
        
        
        
        
        
        
        
        
#        self.main_fig, self.axs = plt.subplots(2,1)
#        self.main_fig.patch.set_facecolor('black') #All background outside plot
#        #Object passed to kv
#        self.main_canvas = FigureCanvasKivyAgg(self.main_fig)
#        self.box1.add_widget(self.main_canvas)
#        
#        
#        #BACKGROUND PLOT
#        self.bkg_twin = self.axs[0]
#        self.bkg_twin.axis([self.a, self.b, self.psi_blim, self.psi_tlim])
#        figheight = self.main_fig.get_figheight() #In inches (100 p = 1 inch)
#        self.bkg_twin.set_title('x [$\AA$]', color = 'white'
#                                , pad = 0.05*figheight*100, fontsize = 10)
#        #pad is passed in points 
#        self.bkg_twin.tick_params(axis = 'x', labelbottom=False,labeltop=True, 
#                                  bottom = False, top = True)
#
#        #We are using their axis for 'x' axis and 'prob' axis for the default
#        #bkg axis position suits us
#        self.bkg_twin.set_ylabel('Probability [$\AA^{-1}$]', color = 'white') 
##        self.bkg_twin.set_xlabel('x [$\AA$]', color = 'white')
#        self.bkg_twin.tick_params(colors='white')
#                
#        #Filling: we take every other border from the file and fill the prev. 
#        # zone(red) and the following zone (green).Last one added a side (red).
#        prev = self.a
#        #To keep track which points are in the red zone
#        self.redzone = np.array([])
#        for i in range(3, len(self.lvl_set)-1, 2):
#            #Index
#            prev_index = int((prev - self.a)//self.deltax)
#            index = int((self.lvl_set[i]-self.a)//self.deltax)
#            nxt_index = int((self.lvl_set[i+1]- self.a)//self.deltax)
#            #Red
#            bot = (np.abs(self.psiev)**2)[prev_index:index+1] #+1 due to slice
#            top = np.zeros_like(bot) + 2.
#            redzone = self.mesh[prev_index:index+1] #+1 due to slice
#            self.redzone = np.append(self.redzone, redzone)
#            self.bkg_twin.fill_between(redzone,bot,top,
#                                       facecolor = self.zonecol_red)
#            #Green
#            bot = (np.abs(self.psiev)**2)[index:nxt_index+1] #(")
#            top = np.zeros_like(bot) + 2.
#            greenzone = self.mesh[index:nxt_index+1] #(")
#            self.bkg_twin.fill_between(greenzone, bot, top,
#                                       facecolor = self.zonecol_green)
#            #Looping by giving the new prev position
#            prev = self.mesh[int((self.lvl_set[i+1]-self.a)//self.deltax)]
#        #Last zone (red)
#        bot = (np.abs(self.psiev)**2)[nxt_index:]
#        top = np.zeros_like(bot) + 2.
#        redzone = self.mesh[nxt_index:]
#        self.redzone = np.append(self.redzone, redzone)
#        self.bkg_twin.fill_between(redzone,bot,top,
#                                       facecolor = self.zonecol_red)
#        
#        
#         #PSI PLOT
#        self.psi_twin = self.bkg_twin.twinx()
#        
#        self.psi_twin.axis([self.a, self.b, self.psi_blim, self.psi_tlim])
#        self.psi_twin.axis('off') #bkg axis already taken care of.
#        
#        self.psi_data, = self.psi_twin.plot(self.mesh, np.abs(self.psi0)**2,
#                                            alpha = 0.0)
#        
#        self.psi_twin.fill_between(self.mesh,np.abs(self.psiev)**2,
#                                       facecolor = 'black')
#        
#          #POTENTIAL PLOT
#        self.pot_twin = self.bkg_twin.twinx()
#        
#        self.pot_twin.axis([self.a, self.b, self.pot_blim, self.pot_tlim])
#        
#        self.pot_twin.set_ylabel('Potential [eV]', color = 'white')
#        self.pot_twin.tick_params(axis='y', colors='white')
#                
#        self.pot_data, = self.pot_twin.plot(self.mesh, self.potential)
#        
#        
#        #VIsualplot
#        self.visuax = self.axs[1]
#        
#        self.visuax.axis([self.a, self.b, -1.0, 1.0])
#        self.visuax.axis('off')
#        
#        for pos in np.linspace(self.a, self.b, 10):
#            self.visuax.add_patch(plt.Circle((pos, 0), 1, color='white', alpha=1))
#        
#        
#         #FIRST DRAW
#        self.main_fig.tight_layout() #Fills all available space (title space).
#        #This 'tight' needs to be at the end so it considers all objects drawn
#        self.main_canvas.draw_idle() 
#        
#        
#        
#        
#        
#        
#        
#        
#        
#        
#        
        
        
        
        
        
        
#        
#        #FIGURE AND CANVAS
#        #Figure where we draw and first subplot 
#        self.main_fig, self.bkg_twin = plt.subplots()
#        self.main_fig.patch.set_facecolor('black') #All background outside plot
#        #Object passed to kv
#        self.main_canvas = FigureCanvasKivyAgg(self.main_fig)
#        self.box1.add_widget(self.main_canvas)
#        
#        #BACKGROUND PLOT
#        self.bkg_twin.axis([self.a, self.b, self.psi_blim, self.psi_tlim])
#        figheight = self.main_fig.get_figheight() #In inches (100 p = 1 inch)
#        self.bkg_twin.set_title('x [$\AA$]', color = 'white'
#                                , pad = 0.05*figheight*100, fontsize = 10)
#        #pad is passed in points 
#        self.bkg_twin.tick_params(axis = 'x', labelbottom=False,labeltop=True, 
#                                  bottom = False, top = True)
#
#        #We are using their axis for 'x' axis and 'prob' axis for the default
#        #bkg axis position suits us
#        self.bkg_twin.set_ylabel('Probability [$\AA^{-1}$]', color = 'white') 
##        self.bkg_twin.set_xlabel('x [$\AA$]', color = 'white')
#        self.bkg_twin.tick_params(colors='white')
#                
#        #Filling: we take every other border from the file and fill the prev. 
#        # zone(red) and the following zone (green).Last one added a side (red).
#        prev = self.a
#        #To keep track which points are in the red zone
#        self.redzone = np.array([])
#        for i in range(3, len(self.lvl_set)-1, 2):
#            #Index
#            prev_index = int((prev - self.a)//self.deltax)
#            index = int((self.lvl_set[i]-self.a)//self.deltax)
#            nxt_index = int((self.lvl_set[i+1]- self.a)//self.deltax)
#            #Red
#            bot = (np.abs(self.psiev)**2)[prev_index:index+1] #+1 due to slice
#            top = np.zeros_like(bot) + 2.
#            redzone = self.mesh[prev_index:index+1] #+1 due to slice
#            self.redzone = np.append(self.redzone, redzone)
#            self.bkg_twin.fill_between(redzone,bot,top,
#                                       facecolor = self.zonecol_red)
#            #Green
#            bot = (np.abs(self.psiev)**2)[index:nxt_index+1] #(")
#            top = np.zeros_like(bot) + 2.
#            greenzone = self.mesh[index:nxt_index+1] #(")
#            self.bkg_twin.fill_between(greenzone, bot, top,
#                                       facecolor = self.zonecol_green)
#            #Looping by giving the new prev position
#            prev = self.mesh[int((self.lvl_set[i+1]-self.a)//self.deltax)]
#        #Last zone (red)
#        bot = (np.abs(self.psiev)**2)[nxt_index:]
#        top = np.zeros_like(bot) + 2.
#        redzone = self.mesh[nxt_index:]
#        self.redzone = np.append(self.redzone, redzone)
#        self.bkg_twin.fill_between(redzone,bot,top,
#                                       facecolor = self.zonecol_red)
#        
#        #PSI PLOT
#        self.psi_twin = self.bkg_twin.twinx()
#        
#        self.psi_twin.axis([self.a, self.b, self.psi_blim, self.psi_tlim])
#        self.psi_twin.axis('off') #bkg axis already taken care of.
#        
#        self.psi_data, = self.psi_twin.plot(self.mesh, np.abs(self.psi0)**2,
#                                            alpha = 0.0)
#        
#        self.psi_twin.fill_between(self.mesh,np.abs(self.psiev)**2,
#                                       facecolor = 'black')
#        
#        
#         #POTENTIAL PLOT
#        self.pot_twin = self.bkg_twin.twinx()
#        
#        self.pot_twin.axis([self.a, self.b, self.pot_blim, self.pot_tlim])
#        
#        self.pot_twin.set_ylabel('Potential [eV]', color = 'white')
#        self.pot_twin.tick_params(axis='y', colors='white')
#                
#        self.pot_data, = self.pot_twin.plot(self.mesh, self.potential)
#        
#        
#        
#        for pos in np.linspace(self.a, self.b, 2):
#            self.pot_twin.add_patch(plt.Circle((pos, -5), 5, color='yellow', alpha=1))
#        
#        
#        
##        
##        #VISUAL REPRESENTATION
##        self.visuax = self.bkg_twin.twinx()
##        
##        self.visuax.axis([self.a, self.b, 0, 50])
##        
#        #Change all background colors inside plots so it seems to be the
#        #outside color
#        self.pot_twin.set_facecolor('black')
#        self.bkg_twin.set_facecolor('black')
#        self.psi_twin.set_facecolor('black')
#        
#        #FIRST DRAW
#        self.main_fig.tight_layout() #Fills all available space (title space).
#        #This 'tight' needs to be at the end so it considers all objects drawn
#        self.main_canvas.draw_idle() 
#        
        
        
        
#        #VISUAL REPRESENTATION PLOT
#        #FIGURE AND CANVAS
#        #Figure where we draw and first subplot 
#        self.visu_fig, self.visuax = plt.subplots()
#        self.visu_fig.patch.set_facecolor('black') #All background outside plot
#        #Object passed to kv
#        self.visu_canvas = FigureCanvasKivyAgg(self.visu_fig)
#        self.box2.add_widget(self.visu_canvas)
#        
#        #VISUAL PLOT
#        self.visuax.axis([self.a, self.b, -1.0, 1.0])
#        self.visuax.axis('off')
#        
#        for pos in np.linspace(self.a, self.b, 10):
#            self.visuax.add_patch(plt.Circle((pos, 0), 1, color='white', alpha=1))
#            
#        self.visu_fig.tight_layout()
#        self.visu_canvas.draw_idle()    
            
        
        
    
        # ------------------------ CLOCK --------------------------------------
        """
        Here all animation will be happening. The plotting function definied in
        the proper section will be called several times per second. It may
        include a pause or play control (the funciton itself here in clock)
        """
        
                                ###### Time steps ######
        #Time variable for the plot (there may be another animation with dif t)
        self.plt_time = 0.
        self.plt_dt = 1./30.
        #Plotting velocity
        self.plt_vel_factor = 9 #Factor in dt
        self.label_vel.text = 'Velocity \n    ' + str(self.plt_vel_factor) +'X'
        #Pausing
        self.pause_state = True #Begins paused
        
                                ###### Clock ######
        Clock.schedule_interval(self.plotpsiev, 1/30.)
        
    #========================== FUNCTIONS =====================================
        
    #___________________"OBJECT KIND OF" FUNCTION _____________________________
    
    #POTENTIAL: trivially enough, the potential of the problem. Its parameters
    #will be obtained from the 'objects' in the app (in the layout) such as 
    #sliders or buttons. This information is passed from the other file .kv 
    #with the identification names of each of the objects there
    def pot_init(self):
        """
        Creates the potential array object, combining the harmonic and the 
        gaussian potential. Same shape as mesh.
        """
        #First line: factor to scale the potential. 
        #Second line and third line: gaussian potential. 
        #Last line: harmonic potential.
        self.potential = 20*(\
                    1./np.sqrt(2*np.pi*self.pot_sigma**2)*\
                    np.exp(-(self.mesh-self.pot_mu)**2/(2.*self.pot_sigma**2))\
                    +\
                    0.5*self.pot_k*self.mesh**2)
        
    
    #PSI0: initial wavefunciton. Later on this will be defined, meaning the 
    #initial conditions by the user or the game itself.
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



    
    #_______________"FUNCTIONAL" FUNCTION (compute something)__________________
    
    #KEYBOARD
    def _keyboard_closed(self):
        """
        Actions taken when keyboard is released/closed. Unbinding and 
        'removing' the _keyboard.
        """
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        print('keyboard released')
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
    
    #CHANGE_VEL: Changes the factor in the time exponential in ary_compexp from
    #plotpsiev().
    def change_vel(self):
        """
        Changes the factor in the plot time diferential.
        """
        self.plt_vel_factor += 2
        if self.plt_vel_factor > 10:
            self.plt_vel_factor = 1
        #Label in kivy file
        self.label_vel.text = 'Velocity \n    ' + str(self.plt_vel_factor) +'X'
        
        
    #PAUSE: Changes the pause state from true to false and viceversa.
    def pause(self):
        """
        Changes the pause state from true to false and viceversa.
        """
        if self.pause_state == True: #Unpause
            self.pause_state = False
            self.label_pause.text = 'Pause'
        else:
            self.pause_state = True #Pause
            self.label_pause.text = 'Play'
    
    #PLOTPSIEV: the one called several times. It includes the calculation of 
    #the new data on the given time (psi(t)), using the given eigenstuff
    #and components (already done by other functions). Also does the 
    #data updating and drawing
    def plotpsiev(self, dt):
        """
        Function to be called in the animation loop (clock.schedule_interval),
        it has to have dt as an argument. Here first is computed psi(t), then
        the data is updated in psi_data and finally it draws on the canvas. 
        This parameter dt is not the actual time, its only the real time 
        interval between calls. The time is obtained with Clock.get_time()
        
        Change: We have to stop using get_time, its a mess later on, and start
        using our own dt.
        """
        #Only evolve plt_time if we are unpaused
        if not self.pause_state:
            #Making one time step here (factor enable us to change 
            #plot velocity)    
            self.plt_time += self.plt_vel_factor*self.plt_dt           
            t = self.plt_time
            #COMPUTE PSIEV(t). We do it with two steps (given t).
            #_1_. Column vector containing the product between component and 
            #exponential factor. First build array 'ary' and 
            #then col. matrix 'mtx'
            ary_compexp = self.comp0 * \
                        np.exp(np.complex(0.,-1.)*self.evals*t/(50*self.hbar))
            mtx_compexp = np.matrix(np.reshape(ary_compexp, (self.N + 1,1)))
            #_2_. Psi(t)
            col_psiev = self.evect * mtx_compexp #Matrix product
            
            #UPDATE DATA. Since col_psiev is a column vector (N+1,1) and we 
            #need to pass a list, we reshape and make it an array.
            self.psiev = np.array(np.reshape(col_psiev, self.N + 1))[0]
            self.psi_data.set_data(self.mesh, np.abs(self.psiev)**2)
            
            #FILLING. Under psi curve in psi_twin & in bck_twin (the zones).
            #Psi
            self.psi_twin.collections.clear()
            self.psi_twin.fill_between(self.mesh, np.abs(self.psiev)**2,
                                       facecolor = 'black')
            #Zones
            self.bkg_twin.collections.clear()
            #We take every other border from the file and fill the prev. zone
            #(red) and the following zone (green).
            prev = self.a
            for i in range(3, len(self.lvl_set)-1, 2):
                #Index
                prev_index = int((prev - self.a)//self.deltax)
                nxt_index = int((self.lvl_set[i+1]- self.a)//self.deltax)
                index = int((self.lvl_set[i]-self.a)//self.deltax)
                #Red
                bot = (np.abs(self.psiev)**2)[prev_index:index+1] #+1 due to slice
                top = np.zeros_like(bot) + 2.
                redzone = self.mesh[prev_index:index+1] #+1 due to slice
                self.bkg_twin.fill_between(redzone,bot,top,
                                           facecolor = self.zonecol_red)
                #Green
                bot = (np.abs(self.psiev)**2)[index:nxt_index+1] #(")
                top = np.zeros_like(bot) + 2.
                greenzone = self.mesh[index:nxt_index+1] #(")
                self.bkg_twin.fill_between(greenzone, bot, top,
                                           facecolor = self.zonecol_green)
                #Looping by giving a new prev position            
                prev = self.mesh[int((self.lvl_set[i+1]-self.a)//self.deltax)]
                
            #Last zone red
            bot = (np.abs(self.psiev)**2)[nxt_index:]
            top = np.zeros_like(bot) + 2.
            redzone = self.mesh[nxt_index:]
            self.bkg_twin.fill_between(redzone,bot,top,
                                           facecolor = self.zonecol_red)
            
            
            
             
            #Visual pltt
            self.visu_im.remove()
            step = int(len(self.psiev)/self.num_visu)
            self.visu_im = self.visuax.imshow([np.abs(self.psiev[::step])**2], 
                          aspect='auto', interpolation = 'gaussian', 
                          cmap = 'gray', alpha = 0.5)
            
            
            
            #VIsualplot
#            [circle.remove() for circle in self.visuax.patches]
#            
#            maxpsi = np.max(np.abs(self.psiev)**2)
#            
#            for i in range(0,len(self.psiev), int(len(self.psiev)/self.num_visu)):
#                prob_psi = np.abs(self.psiev[i])**2
#                a = prob_psi/maxpsi
#                if a > self.min_alpha :
#                    self.visuax.add_patch(plt.Circle((self.mesh[i], 0), 
#                                        self.cir_rad, color='white', alpha=a))



        #Draw (keeps drawing even if ther hasn't been an update)
        self.main_canvas.draw()
    
    #EIGENPARAM: Computes the eigenvalues and eigenvectors of the hamiltonian
    #with a certain potential
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
    
    #COMP: given the initial wave function and the eigenbasis (from eigenparam)
    #gives the components of that wave functioni if that basis.        
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
            
    #MEASURE: triggered from the kivy file. It takes the actual psiev(t), 
    #generates the probability distribution and initiates the psi0 again with a
    #new mu. Finally, calls the comp()
    def measure(self):
        """
        Triggered from the kivy file. It takes the actual psiev(t), generates 
        the probability distribution and picks the new value for mu. A new 
        initial wave function is created with a small sigma witch represents
        the delta post measure. The time needs to be reset to zero after 
        measuring. Finally, calls comp() and now plotpsiev() has everything it
        needs to continue plotting.
        """
        #Get instant probability
        prob = self.deltax*np.abs(self.psiev)**2
        #Pick new random mu0
        self.mu0 = np.random.choice(self.mesh, p=prob)
        #Reset time 
        self.plt_time = 0.
        #Check zone
        #Points
        if self.mu0 in self.redzone: #Bad, out of limits
            self.points -= 5
        else:
            self.points += 10 #level passed
            #Read new lvl
            self.lvl += 1
            self.lvl_set = np.array(eval(self.settings.readline().strip()))
            
            #New redzone and refilling
            self.redzone = np.array([])
            prev = self.a
            for i in range(3, len(self.lvl_set)-1, 2):
                #Index
                prev_index = int((prev - self.a)//self.deltax)
                nxt_index = int((self.lvl_set[-1]- self.a)//self.deltax)
                index = int((self.lvl_set[i]-self.a)//self.deltax)
                
#                #Red
#                bot = (np.abs(self.psiev)**2)[prev_index:index+1] #+1 due to slice
#                top = np.zeros_like(bot) + 2.
                
                #Slicing
                redzone = self.mesh[prev_index:index+1] #+1 due to slice
                
#                self.bkg_twin.fill_between(redzone,bot,top,
#                                           facecolor = self.zonecol_red)
                
                self.redzone = np.append(self.redzone, redzone)
                
#                #Green
#                bot = (np.abs(self.psiev)**2)[index:nxt_index+1] #(")
#                top = np.zeros_like(bot) + 2.
#                greenzone = self.mesh[index:nxt_index+1] #(")
#                self.bkg_twin.fill_between(greenzone, bot, top,
#                                           facecolor = self.zonecol_green)
                
                #Looping by giving the new prev position
                prev = self.mesh[int((self.lvl_set[i+1]-self.a)//self.deltax)]
                
            #Last zone red
            redzone = self.mesh[nxt_index:]
            self.redzone = np.append(self.redzone, redzone)
                
#            bot = (np.abs(self.psiev)**2)[nxt_index:]
#            top = np.zeros_like(bot) + 2.
#            self.bkg_twin.fill_between(redzone,bot,top,
#                                           facecolor = self.zonecol_red)
                
                
#                 for i in range(3, len(self.lvl_set)-1, 2):
#                #Index
#                prev_index = int((prev - self.a)//self.deltax)
#                nxt_index = int((self.lvl_set[i+1]- self.a)//self.deltax)
#                index = int((self.lvl_set[i]-self.a)//self.deltax)
#                #Red
#                bot = (np.abs(self.psiev)**2)[prev_index:index+1] #+1 due to slice
#                top = np.zeros_like(bot) + 2.
#                redzone = self.mesh[prev_index:index+1] #+1 due to slice
#                self.bkg_twin.fill_between(redzone,bot,top,
#                                           facecolor = self.zonecol_red)
#                #Green
#                bot = (np.abs(self.psiev)**2)[index:nxt_index+1] #(")
#                top = np.zeros_like(bot) + 2.
#                greenzone = self.mesh[index:nxt_index+1] #(")
#                self.bkg_twin.fill_between(greenzone, bot, top,
#                                           facecolor = self.zonecol_green)
#                #Looping by giving a new prev position            
#                prev = self.mesh[int((self.lvl_set[i+1]-self.a)//self.deltax)]
#                
#            #Last zone red
#            bot = (np.abs(self.psiev)**2)[nxt_index:]
#            top = np.zeros_like(bot) + 2.
#            redzone = self.mesh[nxt_index:]
#            self.bkg_twin.fill_between(redzone,bot,top,
#                                           facecolor = self.zonecol_red)
            
                
                
            #New pot
            self.pot_mu = self.lvl_set[0]
            self.pot_sigma = self.lvl_set[1]
            self.pot_k = self.lvl_set[2]
            self.pot_init()
            self.pot_data.set_data(self.mesh, self.potential)
            #Eigenparam
            self.eigenparam()
        
        #New psi0
        self.sigma0 = self.dirac_sigma
        self.psi0_init()
        #New comp
        self.comp()
        #Update labels
            #Updating the text in the kivy's labels (these variables come from the 
            #kivy file).
            #Prob info
        self.label1.text = 'New mu0:   ' + '%.1f' % self.mu0 + '\n' \
                           'Prob.:          ' + '%.2f' \
            %(prob[int((self.mu0 - self.a)/self.deltax)] * 100.)\
            + '\n' + \
                           'Max. prob.: ' + '%.2f' \
            %(np.max(prob) * 100.)
            #Compute and show energy's expected value
        energy = np.sum(np.abs(self.comp0)**2 * self.evals)
        self.label2.text = '<E> =  \n' + '%.3f' % energy
            #Game info
        self.label_lvl.text = 'Level ' + str(self.lvl)
        self.label_points.text = str(self.points) + ' points'
        
        
        
        
        #Drawing by the measure function itself (just once). pot already
        #updated.
#        self.psi_data.set_data(self.mesh, np.abs(self.psi0)**2)
#        #FILLING. Under psi curve in psi_twin & in bck_twin (the zones).
#        #Psi
#        self.psi_twin.collections.clear()
#        self.psi_twin.fill_between(self.mesh, np.abs(self.psi0)**2,
#                                   facecolor = 'black')

#        self.main_canvas.draw_idle() 
        
        
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