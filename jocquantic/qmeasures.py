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
from matplotlib.figure import Figure #This figure is the tipical one from 
#matplotlib and is the one we shall 'adapt' to kivy using FigureCanvasKivyAgg
#import matplotlib.pyplot as plt #Testing plots

#import timeit as ti #Used to check run times
from scipy import linalg as spLA #Its attribute eigh_tridiagonal diagonalizes H

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
        
                            ###### DISCRETIZATION ######
        self.a = -10.   
        self.b = 10.        
        self.N = 800
        self.deltax = (self.b - self.a)/float(self.N)
        self.mesh = np.linspace(self.a, self.b, self.N + 1)
        
                        ###### Particle's properties ######
                            
        self.hbar = 0.6582   #In these general units
        
        self.m_elec = 0.1316 #Its the m factor explained in eigenparam function
        self.m = self.m_elec #The name 'm' is the one used inside of eigenparam
        
                            ###### AUXILIAR POTENTIAL ######
        
                    ###### Double well potential's properties ######
                    
        self.pot_mu = 0.
        self.pot_sigma = 0.3
        self.pot_k = 0.4
        #Array with the actual potential values
        self.potential = np.zeros(self.N + 1)
        self.pot_init()
                
                    ###### Eigenvalues and eigenvectors ######
                    
        self.evals = np.zeros(self.N + 1) #Row (N+1) vector (due to eigh func)
        self.evect = np.zeros((self.N + 1, self.N + 1)) #(N+1)x(N+1) matrix
        
                            ###### Wave functions ######
                            
        #Initial (gaussian parameters)
        self.p0 = 0.    
        self.sigma0 = 0.3
        self.mu0 = 0.2
        #Related object
        self.psi0 = np.zeros(self.N + 1)    #Value of the initial wave function
        self.comp0 = np.zeros(self.N + 1)   #Its components. They are row vects
        #First creation of the wave function
        self.psi0_init()  
        #Evolved
        self.psiev = np.zeros(self.N + 1)
        self.compev = np.zeros(self.N + 1)
        #Truncated
        self.trk_psiev = np.zeros(self.N + 1)
        
                                ###### Time steps ######
                                
        #Reset time after measuring (updated after each measure)
        self.rst_time = 0.
        
                                 ###### Measure ######
                                 
        #'Sigma' of dirac's delta after measuring
        self.dirac_sigma = 0.2
        
                                ###### TEXT LABEL ######
                                
        #Updating the text in the kivy's labels (these variables come from the 
        #kivy file).
        self.label1.text = 'New mu0:   ' + '%.1f' % self.mu0 + '\n' + \
            'Prob.:             ' + '-' + '\n' + \
            'Max. prob.:    ' + '-'
            
            
                              ###### TRUNCATING ######
                              
        self.trk_maxeval = 200.
#        trk_start = ti.default_timer()
        self.trk_eparam()
        self.trk_comp()
#        trk_stop = ti.default_timer()
#        print('TRK:  ', trk_stop - trk_start)
        
        

        # ------------ FIRST RUN OF EIGENPARAM AND COMP -----------------------
        
#        start = ti.default_timer()
        self.eigenparam()
        self.comp()  
#        stop = ti.default_timer()
#        print('ALL:  ', stop - start)
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
        
        #Canvas
        self.main_fig = Figure()    #Main ploting object where we will do two 
        #differents plottings ('twins'), one for psi and one for the potential.
        #111 specifies one on top of the other
        #Pot plot
        self.pot_twin = self.main_fig.add_subplot(111)
        self.pot_twin.axis([self.a, self.b, 0, 50])
        self.pot_twin.set_xlabel('x [$\AA$]')
        self.pot_twin.set_ylabel('Potential [eV]')
        #Psi plot
        self.psi_twin = self.pot_twin.twinx()
        self.psi_twin.axis([self.a, self.b, 0, 1.5])
        self.psi_twin.set_ylabel('Probability [$\AA^{-1}$]')
        
#        self.trk_psi_twin = self.pot_twin.twinx()
        
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig)
        
        #Data
        self.pot_data, = self.pot_twin.plot(self.mesh, self.potential)
        self.psi_data, = self.psi_twin.plot(self.mesh, self.psi0)
        #Trk plot in psi's subplot
        self.trk_psi_data, = self.psi_twin.plot(self.mesh, self.psi0) 
        
        #Canvas object in kivy file
        self.box1.add_widget(self.main_canvas)

        # ------------------------ CLOCK --------------------------------------
        """
        Here all animation will be happening. The plotting function definied in
        the proper section will be called several times per second. It may
        include a pause or play control (the funciton itself or here in clock)
        """
        #Clock.schedule_interval(self.plotpsiev, 1/30.)
        #Clock.schedule_interval(self.trk_plotpsiev, 1/30.)
        Clock.schedule_interval(self.all_plotpsiev, 1/30.)
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
        """
        #This t is the time from the beggining of the evolution, when psi(t)
        #was psi0, so everytime we change psiev to psi0 again (measuring), 
        #we have to restart the time.
        t = Clock.get_time() - self.rst_time
        
        #COMPUTE PSIEV(t). We do it with two steps.
        #_1_. Column vector containing the product between component and 
        #exponential factor. First build array 'ary' and then col. matrix 'mtx'
        ary_compexp = self.comp0 * \
                        np.exp(np.complex(0., -1.)*self.evals*t/(50*self.hbar))
        mtx_compexp = np.matrix(np.reshape(ary_compexp, (self.N + 1,1)))
        #_2_. Psi(t)
        col_psiev = self.evect * mtx_compexp #Matrix product
        
        #UPDATE DATA. Since col_psiev is a column vector (N+1,1) and we need to 
        #pass a list, we reshape and make it an array.
        self.psiev = np.array(np.reshape(col_psiev, self.N + 1))[0]
        self.psi_data.set_data(self.mesh, np.abs(self.psiev)**2)
        
        #Norm
#        pp = np.abs(self.psiev)**2
#        nn = self.deltax*(np.sum(pp) - pp[0]/2. - pp[-1]/2.)
#        print(nn, self.deltax*np.sum(pp))
        
        #Draw
        self.main_canvas.draw_idle()  
    
    
    #PLOTPSIEV: the one called several times. It includes the calculation of 
    #the new data on the given time (psi(t)), using the given eigenstuff
    #and components (already done by other functions). Also does the 
    #data updating and drawing

    def trk_plotpsiev(self, dt):
        """
        Function to be called in the animation loop (clock.schedule_interval),
        it has to have dt as an argument. Here first is computed psi(t), then
        the data is updated in psi_data and finally it draws on the canvas. 
        This parameter dt is not the actual time, its only the real time 
        interval between calls. The time is obtained with Clock.get_time() 
        """
        #This t is the time from the beggining of the evolution, when psi(t)
        #was psi0, so everytime we change psiev to psi0 again (measuring), 
        #we have to restart the time.
        t = Clock.get_time() - self.rst_time
        
        #COMPUTE PSIEV(t). We do it with two steps.
        #_1_. Column vector containing the product between component and 
        #exponential factor. First build array 'ary' and then col. matrix 'mtx'
        ary_compexp = self.trk_comp0 * \
                        np.exp(np.complex(0., -1.)*self.trk_evals*t/(50*self.hbar))
        mtx_compexp = np.matrix(np.reshape(ary_compexp, (self.trk_N + 1,1)))
        #_2_. Psi(t)
        col_psiev = self.trk_evect * mtx_compexp #Matrix product
        
        #UPDATE DATA. Since col_psiev is a column vector (N+1,1) and we need to 
        #pass a list, we take out every element individually with a loop. We
        #build the acutal psiev object. Be careful, col_psiev is a matrix and 
        #thus it has an 'extra' index for taking out the value than a normal
        #array would have.
        self.psiev = np.array([psielem[0,0] for psielem in col_psiev])
        self.psi_data.set_data(self.mesh, np.abs(self.psiev)**2)
        
        
        #Norm
#        pp = np.abs(self.psiev)**2
#        nn = self.deltax*(np.sum(pp) - pp[0]/2. - pp[-1]/2.)
#        print(nn, self.deltax*np.sum(pp))
     
        #Draw
        self.main_canvas.draw()  
    
    #PLOTPSIEV: the one called several times. It includes the calculation of 
    #the new data on the given time (psi(t)), using the given eigenstuff
    #and components (already done by other functions). Also does the 
    #data updating and drawing
    def all_plotpsiev(self, dt):
        """
        Function to be called in the animation loop (clock.schedule_interval),
        it has to have dt as an argument. Here first is computed psi(t), then
        the data is updated in psi_data and finally it draws on the canvas. 
        This parameter dt is not the actual time, its only the real time 
        interval between calls. The time is obtained with Clock.get_time() 
        """
        #This t is the time from the beggining of the evolution, when psi(t)
        #was psi0, so everytime we change psiev to psi0 again (measuring), 
        #we have to restart the time.
        t = Clock.get_time() - self.rst_time
        
        #COMPUTE PSIEV(t). We do it with two steps.
        #_1_. Column vector containing the product between component and 
        #exponential factor. First build array 'ary' and then col. matrix 'mtx'
        ary_compexp = self.comp0 * \
                        np.exp(np.complex(0., -1.)*self.evals*t/(50*self.hbar))
        mtx_compexp = np.matrix(np.reshape(ary_compexp, (self.N + 1,1)))
        #_2_. Psi(t)
        col_psiev = self.evect * mtx_compexp #Matrix product
        
        #UPDATE DATA. Since col_psiev is a column vector (N+1,1) and we need to 
        #pass a list, we take out every element individually with a loop. We
        #build the acutal psiev object. Be careful, col_psiev is a matrix and 
        #thus it has an 'extra' index for taking out the value than a normal
        #array would have.
        self.psiev = np.array([psielem[0,0] for psielem in col_psiev])
        self.psi_data.set_data(self.mesh, np.abs(self.psiev)**2)
        
        
        
        #COMPUTE PSIEV(t). We do it with two steps.
        #_1_. Column vector containing the product between component and 
        #exponential factor. First build array 'ary' and then col. matrix 'mtx'
        trk_ary_compexp = self.trk_comp0 * \
                        np.exp(np.complex(0., -1.)*self.trk_evals*t/(50*self.hbar))
        trk_mtx_compexp = np.matrix(np.reshape(trk_ary_compexp, (self.trk_N + 1,1)))
        #_2_. Psi(t)
        trk_col_psiev = self.trk_evect * trk_mtx_compexp #Matrix product
        
        #UPDATE DATA. Since col_psiev is a column vector (N+1,1) and we need to 
        #pass a list, we take out every element individually with a loop. We
        #build the acutal psiev object. Be careful, col_psiev is a matrix and 
        #thus it has an 'extra' index for taking out the value than a normal
        #array would have.
        self.trk_psiev = np.array([psielem[0,0] for psielem in trk_col_psiev])
        self.trk_psi_data.set_data(self.mesh, np.abs(self.trk_psiev)**2 + 0.1)
        
        
        
        
        #Norm
#        pp = np.abs(self.trk_psiev)**2
#        nn = self.deltax*(np.sum(pp) - pp[0]/2. - pp[-1]/2.)
#        print(nn, self.deltax*np.sum(pp))
     
        #Draw
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
        #Check norm
#        pp = np.abs(self.evect)**2
#        nn = self.deltax * (1. - pp[0,:]/2. - pp[-1,:]/2.)
#        print(nn)
        
        #Normalization. Used trapezoids method formula and that sum(evect**2)=1
        factors =1/np.sqrt(self.deltax * \
                               (1. - np.abs(self.evect[0,:])**2/2. - \
                                               np.abs(self.evect[-1,:])**2/2.))
        #Normalized vectors (* here multiplies each factor element by each 
        #evect column)
        self.evect = self.evect * factors
        
        #Check norm again
#        npp = np.abs(self.evect)**2
#        print(np.sum(npp,axis=0))
#        nnn = self.deltax * ( np.sum(npp,axis=0) - npp[0,:]/2. - npp[-1,:]/2.)
#        print(nnn) 
       
        
     #EIGENPARAM: Computes the eigenvalues and eigenvectors of the hamiltonian
    #with a certain potential
    def trk_eparam(self):
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
        
        We are chaging ittt!!!!! Only will return up to some given eval
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
        self.trk_evals, self.trk_evect = spLA.eigh_tridiagonal(main_diag, 
        semi_diag, select = 'v', select_range = (0., self.trk_maxeval), check_finite = False)
        
        self.trk_N = len(self.trk_evals)- 1 #So this new N its equiv to old N
        print(self.trk_N)
        #Check norm
#        pp = np.abs(self.evect)**2
#        nn = self.deltax * (1. - pp[0,:]/2. - pp[-1,:]/2.)
#        print(nn)
        
        #Normalization. Used trapezoids method formula and that sum(evect**2)=1
        trk_factors =1/np.sqrt(self.deltax * \
                               (1. - np.abs(self.trk_evect[0,:])**2/2. - \
                                           np.abs(self.trk_evect[-1,:])**2/2.))
        #Normalized vectors (* here multiplies each factor element by each 
        #evect column)
        self.trk_evect = self.trk_evect * trk_factors
        
        #Check norm again
#        npp = np.abs(self.evect)**2
#        print(np.sum(npp,axis=0))
#        nnn = self.deltax * ( np.sum(npp,axis=0) - npp[0,:]/2. - npp[-1,:]/2.)
#        print(nnn) 
    
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
            
            #Check if we get back the same wave adding up
#            A = np.sum(self.evect * self.comp0, axis = 1)
#            print(A - self.psi0)
       
    #COMP: given the initial wave function and the eigenbasis (from eigenparam)
    #gives the components of that wave functioni if that basis.        
    def trk_comp(self, do_psi = False):
        
        """
        Generates the initial wave function's components on the eigenbasis if 
        d0_psi = False (stored on comp0). If d0_psi = True computes psiev 
        components (stored in compev).
        """
        if do_psi == True:
            
    		#Compute psi's components
            phipsi=np.transpose(np.transpose(np.conj(self.trk_evect)) * self.trk_psiev)
            self.trk_compev = self.deltax * (np.sum(phipsi, axis = 0) \
                                            - phipsi[0,:]/2. - phipsi[-1,:]/2.)
		
        else:
    		#Compute psi0 components
            phipsi=np.transpose(np.transpose(np.conj(self.trk_evect)) * self.psi0)
            self.trk_comp0 = self.deltax * (np.sum(phipsi, axis = 0) \
                                            - phipsi[0,:]/2. - phipsi[-1,:]/2.)
            
            A = np.sum(self.trk_evect * self.trk_comp0, axis = 1)
            pA = np.abs(A)**2
            self.trk_comp0 *= 1. / np.sqrt((self.deltax * (np.sum(pA) - pA[0]/2. - pA[-1]/2.)))
            
            #Check if we get back the same wave adding up
#            print(A - self.psi0)
            
  
        
        
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
        prob = self.deltax*np.abs(self.psiev)**2
        
        self.mu0 = np.random.choice(self.mesh, p=prob)
        self.sigma0 = self.dirac_sigma 
        self.psi0_init()
        self.comp()
        self.rst_time = Clock.get_time()
        #Updating the text in the kivy's labels (these variables come from the 
        #kivy file).
        self.label1.text = 'New mu0:   ' + '%.1f' % self.mu0 + '\n' \
                           'Prob.:          ' + '%.2f' \
            %(prob[int((self.mu0 - self.a)/self.deltax)] * 100.)\
            + '%' + '\n' + \
                           'Max. prob.: ' + '%.2f' \
            %(np.max(prob) * 100.) + '%'
        #Compute and show energy's expected value
        energy = np.sum(np.abs(self.comp0)**2 * self.evals)
        self.label2.text = '<E> =  \n' + '%.3f' % energy
     
    #MEASURE: triggered from the kivy file. It takes the actual psiev(t), 
    #generates the probability distribution and initiates the psi0 again with a
    #new mu. Finally, calls the comp()
    def trk_measure(self):
        """
        Triggered from the kivy file. It takes the actual psiev(t), generates 
        the probability distribution and picks the new value for mu. A new 
        initial wave function is created with a small sigma witch represents
        the delta post measure. The time needs to be reset to zero after 
        measuring. Finally, calls comp() and now plotpsiev() has everything it
        needs to continue plotting.
        """
        prob = self.deltax*np.abs(self.trk_psiev)**2
        prob *= 1./np.sum(prob) #When truncated normalization is not that precise
        self.mu0 = np.random.choice(self.mesh, p=prob)
        self.sigma0 = self.dirac_sigma 
        self.psi0_init()
        self.trk_comp()
        self.rst_time = Clock.get_time()
        #Updating the text in the kivy's labels (these variables come from the 
        #kivy file).
        self.label1.text = 'New mu0:   ' + '%.1f' % self.mu0 + '\n' \
                           'Prob.:          ' + '%.2f' \
            %(prob[int((self.mu0 - self.a)/self.deltax)] * 100.)\
            + '%' + '\n' + \
                           'Max. prob.: ' + '%.2f' \
            %(np.max(prob) * 100.) + '%'
        #Compute and show energy's expected value
        energy = np.sum(np.abs(self.trk_comp0)**2 * self.trk_evals)
        self.label2.text = '<E> =  \n' + '%.3f' % energy

    #MEASURE: triggered from the kivy file. It takes the actual psiev(t), 
    #generates the probability distribution and initiates the psi0 again with a
    #new mu. Finally, calls the comp()
    def all_measure(self):
        """
        Triggered from the kivy file. It takes the actual psiev(t), generates 
        the probability distribution and picks the new value for mu. A new 
        initial wave function is created with a small sigma witch represents
        the delta post measure. The time needs to be reset to zero after 
        measuring. Finally, calls comp() and now plotpsiev() has everything it
        needs to continue plotting.
        """
        prob = self.deltax*np.abs(self.trk_psiev)**2
        prob *= 1./np.sum(prob) #When truncated normalization is not that precise
        self.mu0 = np.random.choice(self.mesh, p=prob)
        self.sigma0 = self.dirac_sigma 
        self.psi0_init()
#        trk_start = ti.default_timer()
        self.trk_comp()
#        trk_stop = ti.default_timer()
#        start = ti.default_timer()
        self.comp()
#        stop = ti.default_timer()
#        print('TRK comp:  ', trk_stop - trk_start)
#        print('ALL comp:  ', stop - start)
        self.rst_time = Clock.get_time()
        #Updating the text in the kivy's labels (these variables come from the 
        #kivy file).
        self.label1.text = 'New mu0:   ' + '%.1f' % self.mu0 + '\n' \
                           'Prob.:          ' + '%.2f' \
            %(prob[int((self.mu0 - self.a)/self.deltax)] * 100.)\
            + '%' + '\n' + \
                           'Max. prob.: ' + '%.2f' \
            %(np.max(prob) * 100.) + '%'
        #Compute and show energy's expected value
        energy = np.sum(np.abs(self.trk_comp0)**2 * self.trk_evals)
        self.label2.text = '<E> =  \n' + '%.3f' % energy

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