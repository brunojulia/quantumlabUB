"""
Jan Albert Iglesias  28/02/2018
"""


import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib
import random as rand

"Parameters"
d=5. #Distance between the peaks.
sigma=0.5 #Standard deviation.
L=10. #Width of the histogram.
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

class Application(tk.Tk):

    def __init__(self):        #Initialization.
        tk.Tk.__init__(self)

        self.title("Schrödinger's cat")

        self.xdat=[]           #List with the measured values.

        self.PDFbutton = tk.Button(self, text="Show PDF", command=lambda: self.showPDF(canvas))
        self.PDFbutton.grid(row=1, column=0)                    #Lambdafuntion allows us to pass arguments to the function.

        self.measurebutton = tk.Button(self, text="Measure!", command=lambda: self.measure(canvas,1))
        self.measurebutton.grid(row=0, column=0) #grid() defines the element's position.

        self.measure10button = tk.Button(self, text="Measure x10", command=lambda: self.measure(canvas,10))
        self.measure10button.grid(row=0, column=1) #pack() can be used instead of grid(), but never use both of them at the same time.

        self.measure100button = tk.Button(self, text="Measure x100", command=lambda: self.measure(canvas,100))
        self.measure100button.grid(row=0, column=2)

        self.clearbutton = tk.Button(self, text="Clear", command=lambda: self.clearall(canvas))
        self.clearbutton.grid(row=1, column=2)

        #Defining the Canvas where plots will appear.
        canvas = FigureCanvasTkAgg(f, self)
        canvas.get_tk_widget().grid(columnspan=3, row=3, column=0) #Columnspan to put it in more than one columns.

        #Setting plot things that are shared by PDF and histogram.
        a.set_title('Measurements histogram')
        a.set_xlabel('x')
        a.set_ylabel('Frequency')
        a.axis([-L/2., L/2., 0., 1.])

    def showPDF(self, canvas): #IMPROVING: It should clear last drawn PDF but not the histogram everytime.
        rangex = np.arange(-L/2., L/2., 0.005)
        a.plot(rangex, wavefun(rangex), 'b-') #Plotting as usual in matplotlib.
        canvas.show() #Showing the plot.

    def measure(self, canvas, nmeasures):
        #It creates 'nmeasures' new values wollowing the PDF wavefun.
        for n in range(nmeasures):
            self.xdat.append(acre(-L/2.,L/2.,M,wavefun))

        histo(self.xdat)

        #Needs to be specified again, for 'histo' clears everything that had been drawn.
        a.set_title('Measurements histogram')
        a.set_xlabel('x')
        a.set_ylabel('Frequency')
        a.axis([-L/2., L/2., 0., 1.])
        canvas.show()

    def clearall(self, canvas):
        self.xdat = []
        a.clear()

        #It draws again the title and labels.
        a.set_title('Measurements histogram')
        a.set_xlabel('x')
        a.set_ylabel('Frequency')
        a.axis([-L/2., L/2., 0., 1.])
        canvas.show()



#Things needed for tk to run.
app = Application()
app.mainloop()
