# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:35:06 2019

@author: rafa

The program follows the nomenclature from:
    Notes on the solution to 1D Schrodinger equation
    for piecewise potentials - bjd
And draws its eigenenergies.

It has 3 sections:
    -WAVE-FUNCTION: computes eigen energies for a given piecewise potential
    -CANVAS: draws both the input potential and the output wave-function
    (gives WAVE-FUNCTION the potential and gets its energy to plot)
    -TOOLS: additional featurings to interact easier with CANVAS,
    or change its properties (such like asking it to ask WAVE-FUNCTION
    to give different energies apart from the ground state)

The grid idea is taken from:
https://stackoverflow.com/questions/52566969/python-mapping-a-2d-array-to-a-grid-with-pyplot

"""

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Button
from matplotlib.ticker import FormatStrFormatter
import numpy as np

# Default number of potential pieces
pieces = 7

data_new = np.zeros((pieces,10),dtype=int)
Vk = np.zeros(pieces,dtype=int)
data = np.copy(data_new)

# Maximum number of potential pieces (buttons won't work for more than 7)
max_pieces = 11

# Default energy level
level = 1

# Maximum number of energy levels to compute
N_root = 5

#-------------------------------------------------------------------------
# WAVE FUNCTION
# Generated following the nomenclature of BJD derivation.

#k_code = k_notes -1
#k_notes = 1 : N-1
#k_code = 0 : N-2
#-------------------------------------------------------------------------


# The energy at the ground state os a square infinite potential
Eguay = np.pi*np.pi*0.5

# Vectors to plot
dx=1/np.float(420)
x_vect=np.arange(0,1+dx,dx)
psi_vect=np.zeros(shape=(len(x_vect)))

# Boundaries position for L=1
xk=np.zeros(pieces)
for k in range(pieces):
    xk[k]=(k+1)/pieces

# We index every piece with the index k, because every 
# piece has its own wave vector k: kk=sqrt(2(E-Vk))
kk=1j*np.zeros(pieces)
    
# phi(k)=Ak*exp(ikkx)+Bk*exp(-ikkx)=[Ak Bk]
phi=1j*np.zeros(shape=(pieces,2))

# Boundary condition for the left wall:
# At x=0 the wave has to be null
# We impose that A0=-B0, and give A0 the arbitrary value 1
# (Arbitrary because the wave will be normalized)
phi[0][0]=1
phi[0][1]=-1

# Continuity condition from piece to piece.
invM=1j*np.zeros(shape=(2,2))
M=1j*np.zeros(shape=(2,2))

# Simpson integration for discrete vectors.
# We use it to normalize the squared wave
def simpson(h,vect):

    #add the extrems
    add=(vect[0]+vect[len(vect)-1])

    #add each parity its factor
    for i in np.arange(2,len(vect)-1,2):
        add+=2*vect[i]

    for i in np.arange(1,len(vect)-1,2):
        add+=4*vect[i]

    #add the global factor
    add*=(h/np.float(3))

    return add 

# Bisection root-finding algorithm
#    
# Improved to recursively:
#    - check both sides of c if needed
#    - check an extra c before discarting an interval
#    - divide the given function by the roots already founded
#        (given the polinomical shape of our function)
#    
# To see how it works, uncomment the prints and try fun(x)=(x-1)(x-2)(x-3).
#
# It doesn't find all the roots. If an interval has 2 roots, but the extra c 
# doesn't fall between them, you will miss them.

def bis(a, b, eps, fun, roots):
    
    def fun_eff(x, fun, roots):
        fx=1
        for xr in roots:
            fx/=x-xr
        fx*=fun(x)
        return fx
    
    fa=fun_eff(a, fun, roots)
    fb=fun_eff(b, fun, roots)
    
    c=0.5*(a+b)
    fc=fun_eff(c, fun, roots)
    
#    print(a,b,c)
    
    if fa==0:
        roots.append(a)
#        print('OK fa==0')
#        print(roots)
#        print('')
#        print('R3? fa==0')
        bis(a+eps, b, eps, fun, roots)
    if fb==0:
        roots.append(b)
#        print('OK fb==0')
#        print(roots)
#        print('')
#        print('L3? fb==0')
        bis(a+eps, b, eps, fun, roots)
    if fc==0:
        roots.append(c)
#        print('OK fc==0')
#        print(roots)
#        print('')
#        print('R3? fc==0')
        bis(c+eps, b, eps, fun, roots)
#        print('L3? fc==0')
        bis(a, c-eps, eps, fun, roots)
        
    if fa*fb<0:
        if abs(a-b)<eps:                
                roots.append(c)
#                print('OK fa*fb<0 & abs(a-b)<eps')
#                print(roots)
#                print('')

        else:
            if fa*fc<0:
#                print('L! fa*fb<0 & fc*fa<0')
                bis(a, c, eps, fun, roots)
#                print('')
#                print('R2? fa*fb<0 & fc*fa<0')
                bis(b, c, eps, fun, roots)
            elif fc*fb<0:
#                print('R! fa*fb<0 & fc*fb<0')
                bis(c, b, eps, fun, roots)
#                print('')
#                print('L2? fa*fb<0 & fc*fa<0')
                bis(a, c, eps, fun, roots)
                
    if fa*fb>0:
        if fa*fc<0:
#            print('L1? fa*fb>0 & fc*fa<0')
            bis(a, c, eps, fun, roots)
        if fc*fb<0:
#            print('R1? fa*fb<0 & fc*fb<0')
            bis(c, b, eps, fun, roots)
    return roots

# Boundary condition for the right wall: At x=1 the wave has to be null
# We compute the left value as a function of the energy. 
# Having imposed that the wave is null on the left wall, only eigen-energies 
# will also null the wave on the right wall. 

def wf_right_wall(E):
    global Vk, pieces, phi, invM, M, kk, xk, Eguay
    
    for k in range(pieces):
        kk[k]=np.sqrt((2+0j)*(E-Vk[k]*Eguay))
    
    # Accumulate the continuity condition from piece to piece 
    # in one effective matrix:
    
    M_eff=np.eye(2)+1j*np.zeros(shape=(2,2))
    
    for k in range(pieces-1):                 
        ex1=np.exp(1j*kk[k]*xk[k])
        ex_1=ex1**-1
        ex2=np.exp(1j*kk[k+1]*xk[k])
        ex_2=ex2**-1
        
        M[0][0]=ex1
        M[0][1]=ex_1
        M[1][0]=1j*kk[k]*ex1
        M[1][1]=-1j*kk[k]*ex_1
        
        invM[0][0]=0.5*ex_2
        invM[0][1]=-0.5j*(1/kk[k+1])*ex_2
        invM[1][0]=0.5*ex2
        invM[1][1]=0.5j*(1/kk[k+1])*ex2
        
        M_eff=np.dot(np.dot(invM,M),M_eff)
    
    # Multiply the last piece of wave by the effective matrix
    
    phi[-1]=np.dot(M_eff,phi[0])
    
    wf_value = phi[-1][0]*np.exp(1j*kk[-1])+phi[-1][1]*np.exp(-1j*kk[-1])
    
    # If the last piece of potential is smaller/bigger than the energy, 
    # the last piece of wave will be real/imaginary. Increasing E, when 
    # E=Vk[-1], the wave makes a transition from 'pure' real to 'pure' 
    # imaginary values. '' because the other part is never 0, as it remains 
    # floating between +- e-16 and not 0. I give one part minus the other 
    # to only cross the axis when one of the parts 'purely' does it.
    # I don't give one part PLUS the other, because at E<Vk[-1] the real part
    # is -ve/+ve and at E>Vk[-1] the imaginary part is +ve/-ve.
    
    return np.real(wf_value)-np.imag(wf_value)

# Computes only the first energy level to inmediate plot
def eigen_energies(N_root_computed):
    global pieces, Vk, E, ixk, psi_vect, Eguay, phi, invM, M, kk, xk, DE, dE
    
    # Step to locate an interval for each root
    DE=Eguay/17
    # Root value precision
    dE=Eguay/1000
    
    # Will return this vector
    E_root=[]
    
    # Start from 
    E=0.99*Eguay
    wave1=wf_right_wall(E)
    
    # Look for the first value
    while len(E_root)<N_root_computed:
        E+=DE
        wave0=wave1
        wave1=wf_right_wall(E)
        
        # If you find an interval, polish it to dE
        if wave0*wave1<0:
            bis(E-DE,E,dE,wf_right_wall,E_root)
    
    # In case you found more than one root, give the first one
    E_root.sort()
    return E_root

# Computes the squared wave function for a given energy E
def psivect():
    global pieces, Vk, E, ixk, psi_vect, Eguay, phi, invM, M, kk, xk, DE, dE
    
    for k in range(pieces):
        kk[k]=np.sqrt((2+0j)*(E-Vk[k]*Eguay))
        
    # Apply in each region the continuity condition to find each piece 
    # of the wave function.
        
    for k in range(pieces-1): 
        #k_python = k_notes -1
        #k_notes = 1 : N-1
        #k_python = 0 : N-2
                        
        ex1=np.exp(1j*kk[k]*xk[k])
        ex_1=ex1**-1
        ex2=np.exp(1j*kk[k+1]*xk[k])
        ex_2=ex2**-1
                
        M[0][0]=ex1
        M[0][1]=ex_1
        M[1][0]=1j*kk[k]*ex1
        M[1][1]=-1j*kk[k]*ex_1
                
        invM[0][0]=0.5*ex_2
        invM[0][1]=-0.5j*(1/kk[k+1])*ex_2
        invM[1][0]=0.5*ex2
        invM[1][1]=0.5j*(1/kk[k+1])*ex2
                
        phi[k+1]=np.dot(np.dot(invM,M),phi[k])
    
    ixk=[0]
    ik=0
    
    # Build the squared wave-function vector from the wave function computed 
    # above and x_vect defined at the begining of the code.
    
    for ix in range(len(x_vect)):
        if x_vect[ix]>xk[ik]:
            ik+=1
            ixk.append(ix-1)
            ixk.append(ix)
    
        ex=np.exp(1j*kk[ik]*x_vect[ix])
            
        psi_vect[ix]=np.absolute(phi[ik][0]*ex+phi[ik][1]*ex**-1)**2

    ixk.append(len(x_vect)-1)
    
    # Normalize it
    psi_vect = np.dot(simpson(dx,psi_vect)**-1,psi_vect)
    
    return psi_vect
    

#-------------------------------------------------------------------------
# CANVAS
# what is being constantly drawn
#-------------------------------------------------------------------------

# Function that draws it all before plotting
def draw_wave(draw_level=1):
    
    global data, cmap, X_hovered, Y_hovered, ax, txt, bricks, left_extrem, right_extrem, top_extrem, bottom_extrem, data_new, ax, x_vect, V_vect, Vk, ixk, psi_vect, E, N_root, eigen_energies_list, level

    ax.cla()
    txt.remove()
    txt = fig.text(0.885, 0.625, 'x %d'%(int(bricks(data))), transform=ax.transAxes)
    
    Vk=data_to_Vk(data)
    
    if draw_level==1:
        E=eigen_energies(1)[0]
        level=1
#        print(level)
    elif draw_level>1 and draw_level<=N_root:
        E=eigen_energies_list[draw_level-1]
    else:
        E=0
    psi_vect=psivect()
    
#    color=['b-', 'g-', 'r-', 'c-', 'm-']
    color=['r']

    for k in range(pieces):
        x = np.dot(pieces,x_vect[ixk[2*k]:ixk[2*k+1]])
        y = np.dot(10/4,psi_vect[ixk[2*k]:ixk[2*k+1]])
        ax.fill(np.append(x, [x[-1], x[0]]),
                np.append(y, [0,0]),
                color[k%len(color)], alpha=0.5)
        ax.plot(x, y, color[k%len(color)])
    
    fig.suptitle(t = '$E_%.i$ = %.3f Eguay'%(draw_level,(E/Eguay)), x = 0.42, y = 0.95)
    ax.set_xticks([piece+0.5 for piece in range(pieces)])
    ax.set_xticklabels([piece+1 for piece in range(pieces)])
    ax.yaxis.set_label_text('V (Eguay)', color='0.5')
    ax.pcolormesh(np.transpose(data[::-1]), cmap=cmap, edgecolors=gridc, linewidths=gridw)
    
    return True

# First draw, when initialized
fig = plt.figure()

# Fake plot below the real one, to put the wave-funtion axis
fake_ax = fig.add_subplot(111)
fake_ax.set_xticks([(1/pieces)*i for i in range(pieces+1)])
fake_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fake_ax.set_xlabel('x/L')
fake_ax.set_yticks([0.1*i for i in range(10+1)])
fake_ax.set_yticklabels([(4*i)/10 for i in range(10+1)])
fake_ax.set_ylabel("$\Psi^2 /A$")

# Real plot where both the potential and the wave are represented
# but only with the potential axis
ax = fake_ax.twiny().twinx()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')  

# colors to plot the matrix 'data' as a relief map: 
#                           background, hover, potential 
cmap = colors.ListedColormap(['white','0.5','0.65'])
# color of the grid
gridc='0.5'
# width of the grid
gridw=1

# recurrent function to connect with the wave-function
def data_to_Vk(data):
    
    global Vk
    
    translated_Vk = np.copy(Vk)
    
    for X in range(len(Vk)):
        translated_Vk[X]=0
        for Y in range(10):
            translated_Vk[X]+=data[len(Vk)-1-X][Y]/2 
        
    return translated_Vk

# Changes the color of the grid and calls draw_wave
def ClickColor(event):
    global data, cmap, X_hovered, Y_hovered, ax, txt, bricks, left_extrem, right_extrem, top_extrem, bottom_extrem, data_new, ax, x_vect, V_vect, Vk, ixk, psi_vect, E
    
    # If the mouse is inside the grid
    if event.x < right_extrem and event.x > left_extrem and event.y < top_extrem and event.y > bottom_extrem: 
        
        #locate the cell
        X=int(event.xdata-event.xdata%1)
        Y=int(event.ydata-event.ydata%1)
        X=(len(data)-1)-X
        
        # hover the cell
        X_hovered=X
        Y_hovered=Y
        
        # save the grid before changing it
        data0=np.copy(data)
        
        # if I don't have enough bricks, draw the remaining only
        if (Y+1)>(bricks(data)+data_to_Vk(data)[pieces-1-X]):
            Y=bricks(data)+data_to_Vk(data)[pieces-1-X]-1
        
        # if V=1 you can delete the brick
        if Y==0 and data_to_Vk(data)[pieces-1-X]==1:
            data[X][:Y+1]=0
        else:
            data[X][:Y+1]=2
        data[X][Y+1:]=0
        
        # do I have enough bricks?
        if bricks(data)<0:
            data=np.copy(data0)
            return True
        else:
            draw_wave()
            event.canvas.draw()

# only changes the color of the grid
def HoverColor(event):
    global data, cmap, X_hovered, Y_hovered, X_hovered, Y_hovered, ax, left_extrem, right_extrem, top_extrem, bottom_extrem, gridc, gridw
    
    # If the mouse is outside the grid
    if event.x > right_extrem or event.x < left_extrem or event.y > top_extrem or event.y < bottom_extrem: 
        
        # dishover
        X_hovered=30
        Y_hovered=30
        
        
        data[data==1]=0

        # draw dishovered grid
        ax.pcolormesh(np.transpose(data[::-1]), cmap=cmap, edgecolors=gridc, linewidths=gridw)
        event.canvas.draw()
        return True

    # If the mouse is inside the grid
    elif isinstance(event.xdata, float) and isinstance(event.ydata, float):

        X=int(event.xdata-event.xdata%1)
        Y=int(event.ydata-event.ydata%1)
        X=(len(data)-1)-X
        
        # don't hover twice
        if X==X_hovered and Y==Y_hovered:
            return True
        
        else:
    
            #hover this
            X_hovered=X
            Y_hovered=Y
    
            data0=np.copy(data)
            
            # only hover the remaining bricks
            if (Y+1)>(bricks(data)+data_to_Vk(data)[pieces-1-X]):
                Y=bricks(data)+data_to_Vk(data)[pieces-1-X]-1
            
            data[X][:Y+1]=1
            
            ax.pcolormesh(np.transpose(data[::-1]), cmap=cmap, edgecolors=gridc, linewidths=gridw)
            event.canvas.draw()
            
            data=data0
            
            
            return True
    else:
        return True

# in case of resizing the window, take all this into account    
axl=0.12
axr=0.72
axt=0.85
axb=0.1

plt.subplots_adjust(left=axl,right=axr,top=axt, bottom=axb)
left_extrem=(fig.get_size_inches()*fig.dpi)[0]*axl+gridw
right_extrem=(fig.get_size_inches()*fig.dpi)[0]*axr-gridw
top_extrem=(fig.get_size_inches()*fig.dpi)[1]*axt-gridw
bottom_extrem=(fig.get_size_inches()*fig.dpi)[1]*axb+gridw

def ResizeExtrems(event):
    global left_extrem, right_extrem, top_extrem, bottom_extrem, axr, axl, axt, axb, gridw
    left_extrem=(fig.get_size_inches()*fig.dpi)[0]*axl+gridw
    right_extrem=(fig.get_size_inches()*fig.dpi)[0]*axr-gridw
    top_extrem=(fig.get_size_inches()*fig.dpi)[1]*axt-gridw
    bottom_extrem=(fig.get_size_inches()*fig.dpi)[1]*axb+gridw

# okey, connect the mouse
cid_click = fig.canvas.mpl_connect('button_press_event', ClickColor)
cid_hover = fig.canvas.mpl_connect('motion_notify_event', HoverColor)
cid_resize = fig.canvas.mpl_connect('resize_event', ResizeExtrems)
    
#---------------------------------------------------------------------------
# TOOLS 
# Buttons and miscelaneous at the left part of the screen.
# From top to bottom
#---------------------------------------------------------------------------

# [ + ] button 
# adds an piece up to max_pieces

def more(event):
    global data, data_new, ax, Vk, pieces, txt, bricks, xk, kk, phi, max_pieces
    
    if pieces<max_pieces:
        
        pieces+=1

        data_new = np.zeros((pieces,10),dtype=int)
        Vk = np.zeros(pieces,dtype=int)
        data = np.copy(data_new)
        
        xk=np.zeros(pieces)
        for k in range(pieces):
            xk[k]=(k+1)/pieces
            
        kk=1j*np.zeros(pieces)
        phi=1j*np.zeros(shape=(pieces,2))
        phi[0][0]=1
        phi[0][1]=-1

        draw_wave()
        
        event.canvas.draw()
        
        return True
        
    else: return True

axmore = plt.axes([0.825, 0.8, 0.05, 0.075])
bmore = Button(axmore, '+')
bmore.on_clicked(more)

#---------------------------------------------------------------------------

# [ - ] button
# substract a piece down to 1

def less(event):
    global data, data_new, ax, Vk, pieces, txt, bricks, xk, kk, phi
    
    if pieces>1:
        
        pieces+=-1

        data_new = np.zeros((pieces,10),dtype=int)
        Vk = np.zeros(pieces,dtype=int)
        data = np.copy(data_new)
        
        xk=np.zeros(pieces)
        for k in range(pieces):
            xk[k]=(k+1)/pieces
            
        kk=1j*np.zeros(pieces)
        phi=1j*np.zeros(shape=(pieces,2))
        phi[0][0]=1
        phi[0][1]=-1

        draw_wave()

        event.canvas.draw()
        
        return True
        
    else: return True

axless = plt.axes([0.875, 0.8, 0.05, 0.075])
bless = Button(axless, '-')
bless.on_clicked(less)

#---------------------------------------------------------------------------

# new botton
# draws the squared infinite potential (nothing)

def new(event):

    global data, data_new
    
    data = np.copy(data_new)
    
    draw_wave()
    event.canvas.draw()

axnew = plt.axes([0.825, 0.7, 0.1, 0.075])
bnew = Button(axnew, 'New')
bnew.on_clicked(new)

#---------------------------------------------------------------------------

# remaining bricks
# Counts how many bricks you have.
# 6*pieces for people to get creative.

def bricks(data):
    global pieces
    
    bricks0 = 6*pieces
    
    return bricks0-np.sum(data_to_Vk(data))

axbrick = plt.axes([0.820, 0.62, 0.05, 0.0375])
bbrick = Button(axbrick, '', color='0.65', hovercolor='0.5')

txt = fig.text(0.885, 0.625, 'x %d'%(int(bricks(data))), transform=ax.transAxes)
    
#---------------------------------------------------------------------------

# [ ↑ ] button
# changes from the first energy level to the others, up to N_root

def up(event):
    global level, N_root, eigen_energies_list
    
    if level==1:
        eigen_energies_list=eigen_energies(N_root)
#        print(eigen_energies_list)
    
    if level>=1 and level<N_root:  
        
        level+=1
#        print(level)
        draw_wave(level)
        event.canvas.draw()
        
        return True
        
    else: return True

axup = plt.axes([0.825, 0.5, 0.05, 0.075])
bup = Button(axup, ' ↑ ')
bup.on_clicked(up)

#---------------------------------------------------------------------------

# [ ↓ ] button
# changes from any level down to the first one

def down(event):
    global level, N_root

    if level>1 and level<=N_root:  
        level-=1
#        print(level)
        draw_wave(level)
        event.canvas.draw()
        
        return True
        
    else: return True

axdown = plt.axes([0.875, 0.5, 0.05, 0.075])
bdown = Button(axdown, ' ↓ ')
bdown.on_clicked(down)

#---------------------------------------------------------------------------

# demo buttons
# demos are drawn by hand for pieces=1..7 only

fig.text(0.838, 0.425, 'demos', transform=ax.transAxes)

def Vk_to_data(Vk):
    
    global data_new
    
    translated_data = np.copy(data_new)
    
    for X in range(len(Vk)):
        Y=Vk[X]-1
        translated_data[X][:Y+1]=2
        translated_data[X][Y+1:]=0
    
    return translated_data


def gauss(event):
    global data, pieces

    Vk=[[0],
        [0,0],[5,1,5],
        [5,1,1,5],[8,2,0,2,8],
        [6,1,0,0,1,6],[9,4,1,0,1,4,9]]

    if pieces<=7:
        data = Vk_to_data(Vk[pieces-1])
        draw_wave()
        event.canvas.draw()
    else:
        return True

axgauss = plt.axes([0.825, 0.325, 0.1, 0.075])
bgauss = Button(axgauss, 'Gauss')
bgauss.on_clicked(gauss)


def step(event):
    global data, pieces

    Vk=[[0],
        [0,9],[0,9,9],
        [0,0,9,9],[0,0,9,9,9],
        [0,0,0,9,9,9],[0,0,0,9,9,9,9]]

    if pieces<=7:
        data = Vk_to_data(Vk[pieces-1])
        draw_wave()
        event.canvas.draw()
    else:
        return True

axstep = plt.axes([0.825, 0.225, 0.1, 0.075])
bstep = Button(axstep, 'Step')
bstep.on_clicked(step)


def wall(event):
    global data, pieces

    Vk=[[0],[0,0],
        [0,10,0],
        [0,10,10,0],
        [0,0,10,0,0],
        [0,0,10,10,0,0],
        [0,0,0,10,0,0,0]]

    if pieces<=7:
        data = Vk_to_data(Vk[pieces-1])
        draw_wave()
        event.canvas.draw()
    else:
        return True

axwall = plt.axes([0.825, 0.125, 0.1, 0.075])
bwall = Button(axwall, 'Wall')
bwall.on_clicked(wall)


#---------------------------------------------------------------------------

# end

draw_wave()

plt.show()





















