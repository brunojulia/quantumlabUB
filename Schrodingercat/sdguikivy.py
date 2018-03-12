"""
Jan Albert Iglesias  12/03/2018
"""


from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.app import App
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg #Needs to be installed: "garden install matplotlib".
from matplotlib.figure import Figure
import matplotlib
import random as rand
import numpy as np

"Parameters"
d=5.         #Distance between the peaks.
sigma=0.5    #Standard deviation.
L=10.        #Width of the histogram.
M=1./(np.sqrt(2.*np.pi)*sigma) #Upper bound of wavefun(x).

def wavefun(x):
    """
    Wavefunction of the particle. It consists of two gaussian functions separated a distance d.
    """
    #It is actually the squared modulus of the particle's wavefunction; its probability density function.
    gleft=np.exp(-(x-d/2.)**2./(2.*sigma**2.))
    gright=np.exp(-(x+d/2.)**2./(2.*sigma**2.))
    wavefun=1./(np.sqrt(8.*np.pi)*sigma)*(gleft+gright) #Normalitzation.
    return wavefun

"Acceptance/Rejection method"
def acre(a,b,M,fun):
    """
    Returns a random number between a and b with a probability density function (PDF) fun(x).
    M is an upper bound of function fun(x).
    """
    #Two random numbers x,p (following uniform distributions) are generated and compared.
    x=rand.uniform(a,b)
    p=rand.uniform(0,M)

    #If fun(x) >= p, number x is accepted as a random number with a PDF fun(x).
    #Otherwise, the process is repeated.
    while fun(x) <= p:
        x=rand.uniform(a,b)
        p=rand.uniform(0,M)

    return x

"Histogram"
def histo(xdat):
    """
    Draws the normalized histogram of the data in xdat (list).
    It creates approximately 2·sqrt(N) boxes, where N is de number of used points.
    The histogram is defined within the limits [-L/2,L/2].
    """
    #It defines 2·sqrt(N) boxes.
    N=len(xdat)
    nboxes=2*int(np.sqrt(N)) #It is always an even number.
    width=L/nboxes

    #It uses two 'counters' in order to plot one in green (living cat) and the other in red (dead cat).
    boxcountleft=np.zeros(int(nboxes/2),dtype=int) #Number of dots in each box.
    boxcountright=np.zeros(int(nboxes/2),dtype=int)
    boxnameleft=np.linspace(-L/2.+width/2.,-width/2.,nboxes/2) #Position of the center of each box.
    boxnameright=np.linspace(width/2.,L/2.-width/2.,nboxes/2)

    #Box indexes range from '0' for the [-L/2,-L/2+width) box to 'nboxes-1' for [L/2-width,L/2].
    for xval in xdat:
        if xval < 0.:
            i = int((xval+L/2.)/width)          #It finds the correct box dividing by width and taking the integer part.
            boxcountleft[i] = boxcountleft[i] + 1
        elif xval >= 0.:
            i = int(xval/width)
            if xval == nboxes/2:                #The rightmost value in the box is added 'manually' since it would fall into another box.
                i = nboxes-1
            boxcountright[i] = boxcountright[i] + 1

    #Normalization.
    boxcountleft=boxcountleft/(N*width)
    boxcountright=boxcountright/(N*width)

    #We use 'a' because it is going to be used in the GUI (graphical user interface), by the canvas subplot.
    a.clear() #Clears everything displayed on the screen.

    #Plotting the histogram in two different colors and the dot marker.
    a.bar(boxnameright, boxcountright, width, edgecolor='black', color='red')
    a.bar(boxnameleft, boxcountleft, width, edgecolor='black', color='green')
    a.plot(xdat[-1], 0., c='black', marker='x', ms=12, mew=2)

    #Showing interesting data.
    a.text(L/4., 0.85, 'N=%d'%N)
    a.text(-L/4., 0.85, 'Last value: x=%f'%xval)
    return

#Plot definitions.
f = Figure(figsize=(6,5), dpi=100)  #Size and resolution.
a = f.add_subplot(111)              #111 to fill all the screen. 121 to use only half screen, etc.


#Kivy GUIs can be created either using only python language or using both python and .kv languages. Only python language is used here.
#When using python, two classes need to be defined. One to set the layout(s) and another to 'run' the application.

#The first class is used to set all the layout (buttons, labels, etc).
class GUIlayout(BoxLayout):
    def __init__(self): #A second argument '**kwargs' can be added here, but not needed.
        super(GUIlayout, self).__init__() #The same here.

        #Two layouts are going to be used. A 'small' one (a GridLayout named 'seclayout') that is going to contain the buttons,
        #and a 'big' one (a BoxLayout referred by 'self') that is going to contain the canvas and the small layout.
        self.orientation = 'vertical'
        self.spacing = (5,5)

        self.xdat=[]           #List with the measured values.

        #The small layout and its parameters.
        seclayout = GridLayout()
        seclayout.rows = 2
        seclayout.cols = 3
        seclayout.size_hint_y = .2 #The percentage of the big layout in the y direction that this layout covers.
        #This is for the canvas to be bigger than the buttons.
        seclayout.row_force_default = True
        seclayout.row_default_height = 50

        #Buttons
        self.measurebutton = Button(text="Measure!")
        self.measurebutton.bind(on_press=lambda x: self.measure(canvas,1,wavefun)) #Lambda x: in order to send arguments to the function and avoid
        seclayout.add_widget(self.measurebutton)                           #"TypeError: <lambda>() takes 0 positional arguments but 1 was given"

        self.measure10button = Button(text="Measure x10")
        self.measure10button.bind(on_press=lambda x: self.measure(canvas,10,wavefun))
        seclayout.add_widget(self.measure10button)

        self.measure100button = Button(text="Measure x100")
        self.measure100button.bind(on_press=lambda x: self.measure(canvas,100,wavefun))
        seclayout.add_widget(self.measure100button)

        self.PDFbutton = Button(text="Check!")
        self.PDFbutton.bind(on_press=lambda x:self.showPDF(canvas))
        seclayout.add_widget(self.PDFbutton)

        self.clearbutton = Button(text="Clear")
        self.clearbutton.bind(on_press=lambda x: self.clearall(canvas))
        seclayout.add_widget(self.clearbutton)

        self.add_widget(seclayout) #The small layout is attached to the big layout as a usual widget.

        canvas = FigureCanvasKivyAgg(f)
        self.add_widget(canvas)
        #self. because it is attached to the big layout.

        #Setting plot things that are shared by PDF and histogram.
        a.set_title('Measurements histogram')
        a.set_xlabel('x')
        a.set_ylabel('Frequency')
        a.axis([-L/2., L/2., 0., 1.])

    #Functions:
    def showPDF(self, canvas): #TO BE IMPROVED: It ought not to overplot the function if it is aleary plotted.
        """
        Plots the Probability Density Function.
        """
        rangex = np.arange(-L/2., L/2., 0.005)
        a.plot(rangex, wavefun(rangex), 'b-') #Plotting as usual in matplotlib.
        canvas.draw() #Showing the plot.

    def measure(self, canvas, nmeasures, PDFfun):
        """
        It creates 'nmeasures' new values following the PDFfun and append them into self.xdat.
        """

        for n in range(nmeasures):
            self.xdat.append(acre(-L/2.,L/2.,M,PDFfun))

        #Then it creates the histogram.
        histo(self.xdat)

        #Plot things need to be specified again, for 'histo' clears everything that had been drawn.
        a.set_title('Measurements histogram')
        a.set_xlabel('x')
        a.set_ylabel('Frequency')
        a.axis([-L/2., L/2., 0., 1.])
        canvas.draw() #kivy uses .daraw() whereas tkinter uses .show().

    def clearall(self, canvas):
        """
        It erases the canvas.
        """

        self.xdat = []
        a.clear()

        #It draws again the title and labels.
        a.set_title('Measurements histogram')
        a.set_xlabel('x')
        a.set_ylabel('Frequency')
        a.axis([-L/2., L/2., 0., 1.])
        canvas.draw()



#The second class is just this. It actually "creates the application".
class GUIApp(App):
    def build(self):
        self.title="Schrödinger's cat"
        return GUIlayout()

#This is used to run the application.
if __name__ == "__main__":
    GUIApp().run()
