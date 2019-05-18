# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:35:06 2019

@author: rafa

The program follows the nomenclature from:
    Notes on the solution to 1D Schrodinger equation
    for piecewise potentials - bjd
And draws its eigenenergies.

It has 4 sections:
    -WAVE-FUNCTION: computes eigen energies for a given piecewise potential
    -CANVAS: draws both the input potential and the output wave-function
    (gives WAVE-FUNCTION the potential and gets its energy to plot)
    -TOOLS: additional featurings to interact easier with CANVAS,
    or change its properties (such like asking it to ask WAVE-FUNCTION
    to give different energies apart from the ground state)
    -SCORE: for the survival mode

The grid idea is taken from:
https://stackoverflow.com/questions/52566969/python-mapping-a-2d-array-to-a-grid-with-pyplot

The annotations are taken from:
https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib

The detrans[late] inversion is from:
https://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping
"""

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Button
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Ellipse
import numpy as np
import random

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

# Survival initial stuff
max_value = 9
score=0
ball_value=9
lives=3
x_ball=0
y_ball=9

X_hovered=30
Y_hovered=30

difficulty=10
no_bricks=False
score_color=(0,1,0)
ball_color=(0,1,0)

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
def draw_wave():
    
    global trans, late, data, cmap, ax, txt, bricks, \
    left_extrem, right_extrem, top_extrem, bottom_extrem, \
    data_new, ax, x_vect, V_vect, Vk, ixk, psi_vect, E, \
    N_root, eigen_energies_list, level, \
    x_ball, y_ball, \
    score, lives, ball_value, value_txt, score_txt, lives_txt,\
    bdiff, axdiff, ldiff, difficulty, score_color, ball_color


    if score>=100:
        if bdiff.label.get_text()=='EASY':
            difficulty=5
            ball_color=(1,1,0)
            bdiff.label.set_text('FAIR.')
            axdiff.spines['top'].set_color((1,1,0))
            axdiff.spines['bottom'].set_color((1,1,0))
            axdiff.spines['right'].set_color((1,1,0))
            axdiff.spines['left'].set_color((1,1,0))
        if bdiff.label.get_text()=='FAIR':
            difficulty=4
            ball_color=(1,0,0)
            bdiff.label.set_text('HARD')
            axdiff.spines['top'].set_color((1,0,0))
            axdiff.spines['bottom'].set_color((1,0,0))
            axdiff.spines['right'].set_color((1,0,0))
            axdiff.spines['left'].set_color((1,0,0))
            
    if score>=200:
        if bdiff.label.get_text()=='FAIR.':
            difficulty=4
            ball_color=(1,0,0)
            bdiff.label.set_text('HARD')
            axdiff.spines['top'].set_color((1,0,0))
            axdiff.spines['bottom'].set_color((1,0,0))
            axdiff.spines['right'].set_color((1,0,0))
            axdiff.spines['left'].set_color((1,0,0))

    ax.cla()

    Vk=data_to_Vk(data)
    
    E=eigen_energies(level)[level-1]
    
#    if draw_level==1:
#        E=eigen_energies(1)[0]
#        level=1
##        print(level)
#    elif draw_level>1 and draw_level<=N_root:
#        E=eigen_energies_list[draw_level-1]
#    else:
#        E=0
    psi_vect=psivect()
    
#    color=['b-', 'g-', 'r-', 'c-', 'm-']
    color=['r']

    touched=np.zeros(pieces,dtype=int)

    for k in range(pieces):
        x = np.dot(pieces,x_vect[ixk[2*k]:ixk[2*k+1]])
        y = np.dot(10/4,psi_vect[ixk[2*k]:ixk[2*k+1]])
        ax.fill(np.append(x, [x[-1], x[0]]),
                np.append(y, [0,0]),
                color[k%len(color)], alpha=(0.5+0.12*(level-1)))
        ax.plot(x, y, color[k%len(color)])
        
        touched[k]=int(max(y)-0.5)
#        print(touched[-1][0]+1,touched[-1][1]+1)
#    print('')

    play_ball(touched)
    ball_the_data()
    
    fig.suptitle(t = '$E %.i$ = %.3f $E_{ini}$'%(level,(E/Eguay)), x = 0.42, y = 0.97, color='r')
    ax.set_xticks([piece+0.5 for piece in range(pieces)])
    ax.set_xticklabels([piece+1 for piece in range(pieces)])
    ax.yaxis.set_label_text('V ($E_{ini}$)')
    ax.spines['top'].set_color(gridc)
    ax.spines['top'].set_linewidth(2*gridw)
    ax.pcolormesh(np.transpose(data[::-1]), cmap=cmap(data), edgecolors=gridc, linewidths=gridw)
    ax.axis([0,pieces,0,10])
    
    unball_the_data()
    
    if lives==0:
        fig.canvas.mpl_disconnect(cid_click)
        lives_txt.remove()    
        lives_txt = fig.text(0.815, 0.815, trans[late]['game over'])#, transform=ax.transAxes)
        score_txt.remove()    
        score_txt = fig.text(0.82, 0.765, trans[late]['score = %d']%(score),bbox={'facecolor':score_color, 'alpha':0.5, 'pad':5})#, transform=ax.transAxes)
    else:
        lives_txt.remove()    
        lives_txt = fig.text(0.82, 0.815, '<3 '*lives)#, transform=ax.transAxes)
        score_txt.remove()    
        score_txt = fig.text(0.82, 0.765, trans[late]['score = %d']%(score)*(int(lives/abs(lives))) ,bbox={'facecolor':score_color, 'alpha':0.5, 'pad':5})#, transform=ax.transAxes)
        value_txt.remove()
        value_txt = fig.text(axl+(axr-axl)*(1-(x_ball+0.675)/pieces),\
                             axb+(axt-axb)*((y_ball+0.35)/10), \
                             '+%d'%(ball_value)*(int(lives/abs(lives))))#, transform=ax.transAxes)
    
    return True

# in case of resizing the window, take all this into account    
axl=0.12
axr=0.72
axt=0.85
axb=0.1

# First draw, when initialized
fig = plt.figure()
fig.patch.set_facecolor('0.65')

# Upper white rectangle
upper_ax = plt.Axes(fig, [axl, axt, axr-axl, 1-axt], )
fig.add_axes(upper_ax)
upper_ax.tick_params(color='0.65', labelcolor='0.65')
upper_ax.spines['top'].set_color('w')

# Fake plot below the real one, to put the wave-funtion axis
fake_ax = fig.add_subplot(111)
fake_ax.set_xticks([(1/pieces)*i for i in range(pieces+1)])
fake_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fake_ax.set_xlabel('x/L')
fake_ax.set_yticks([0.1*i for i in range(10+1)])
fake_ax.set_yticklabels([(4*i)/10 for i in range(10+1)])
fake_ax.set_ylabel('$\Psi^2 /A$')

# Real plot where both the potential and the wave are represented
# but only with the potential axis. (The rest is drawn on draw())
ax = fake_ax.twiny().twinx()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right') 

# colors to plot the matrix 'data' as a relief map: 
#                           background, hover, potential 
def cmap(data):
    global ball_value, ball_color
    try:
        ball_value
    
    except NameError: #The game just started
        ball_value = max_value

    if bmode.label.get_text()=='SURVIVAL':
        value_color=(max(ball_color[0],0.7*(1-ball_value/(max_value))),
                    max(ball_color[1],0.7*(1-ball_value/(max_value))),
                    max(ball_color[2],0.7*(1-ball_value/(max_value))))

        cmap = colors.ListedColormap(['white','0.5','0.65',value_color])
        if data.min()==1:
            cmap = colors.ListedColormap(['0.5','0.65', value_color])
        if data.min()==2:
            cmap = colors.ListedColormap(['0.65', value_color])
    else:
        cmap = colors.ListedColormap(['white','0.5','0.65'])
        if data.min()==1:
            cmap = colors.ListedColormap(['0.5','0.65'])
        if data.min()==2:
            cmap = colors.ListedColormap(['0.65'])
    return cmap
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
    global data, cmap, X_hovered, Y_hovered, ax, txt, bricks, \
    left_extrem, right_extrem, top_extrem, bottom_extrem, \
    data_new, ax, x_vect, V_vect, Vk, ixk, psi_vect, E, \
    lives, score, score_txt, lives_txt,\
    level, no_bricks
    
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
#        if (Y+1)>(bricks(data)+data_to_Vk(data)[pieces-1-X]):
#            print('you try',(Y+1),'when',data_to_Vk(data)[pieces-1-X])
#            print('can give',(bricks(data)+data_to_Vk(data)[pieces-1-X]))
#            Y=bricks(data)+data_to_Vk(data)[pieces-1-X]-1
#            if Y==-1:
#                no_bricks=True
#                print(X,Y,'no bricks',bricks(data))

#        print('------------------------------')
#        print(Y+1,'-',data_to_Vk(data)[pieces-1-X],'=<',bricks(data),
#              (Y+1-data_to_Vk(data)[pieces-1-X])<=bricks(data))
#        #if I don't have enough bricks, erase and draw the remaining only
#        if (Y+1)>(bricks(data)+data_to_Vk(data)[pieces-1-X]):
#            print('you try',(Y+1),'when',data_to_Vk(data)[pieces-1-X])
#            if (Y+1)<data_to_Vk(data)[pieces-1-X]:
#                print('okey u erasing')
#                no_bricks=False
#            else:
#                print('you have no bricks')

        print(bricks(data))
        #only give the remaining bricks
        if bricks(data)>0 and \
           (Y+1)>(bricks(data)+data_to_Vk(data)[pieces-1-X]):
            Y=bricks(data)+data_to_Vk(data)[pieces-1-X]-1
            print('te doy menos')
        #or if you are erasing
        elif bricks(data)<=0:
            if (Y+1)<=data_to_Vk(data)[pieces-1-X]:
                Y=Y_hovered
                no_bricks=False
                print('vale, borrando')
            else:
                no_bricks=True
                print('click invalido')

        # if V=click you can delete the entire column
        if Y+1==data_to_Vk(data)[pieces-1-X]:
            data[X][:Y+1]=0
        else:
            data[X][:Y+1]=2
        data[X][Y+1:]=0 
        
#        # if V=1 you can delete the brick
#        if Y==0 and data_to_Vk(data)[pieces-1-X]==1:
#            data[X][:Y+1]=0
#        else:
#            data[X][:Y+1]=2
#        data[X][Y+1:]=0
        
        # do I have enough bricks?
        if no_bricks:
            print('no bricks')
            data=np.copy(data0)
            return True
        else:
            draw_wave()
            event.canvas.draw()

# only changes the color of the grid
def HoverColor(event):
    global trans, late, data, cmap, X_hovered, Y_hovered, X_hovered, Y_hovered, \
    ax, left_extrem, right_extrem, top_extrem, bottom_extrem,  \
    gridc, gridw, \
    binfo, axinfo, linfo, axmore, lmore, axless, lless, \
    axnew, lnew, axmode, lmode, axup, lup, axdown, ldown, \
    axgauss, lgauss, axstep, lstep, axwall, lwall
    
    # If the mouse is outside the grid
    if event.x > right_extrem or event.x < left_extrem or event.y > top_extrem or event.y < bottom_extrem: 
        
        # dishover
        X_hovered=30
        Y_hovered=30
        
        
        data[data==1]=0

        # draw dishovered grid
        ball_the_data()
        ax.pcolormesh(np.transpose(data[::-1]), cmap=cmap(data), edgecolors=gridc, linewidths=gridw)
        event.canvas.draw()
        unball_the_data()  
        
        #Also label all buttons the mouse passes by
        
        if event.inaxes == axinfo:
            if not linfo.get_visible():
                linfo.set_visible(True)
        else:
            linfo.set_visible(False)
        
        if binfo.label.get_text()==trans[late]['Instructions: on']:
            
            if axdiff.get_visible():
                if event.inaxes == axdiff:
                    if not  ldiff.get_visible():
                        ldiff.set_visible(True)
                else:
                    ldiff.set_visible(False)
    
            if axmore.get_visible():
                if event.inaxes == axmore:
                    if not lmore.get_visible():
                        lmore.set_visible(True)
                else:
                    lmore.set_visible(False)
                    
            if axless.get_visible():
                if event.inaxes == axless:
                    if not lless.get_visible():
                        lless.set_visible(True)
                else:
                    lless.set_visible(False)
                    
            if axnew.get_visible():
                if event.inaxes == axnew:
                    if not lnew.get_visible():
                        lnew.set_visible(True)
                else:
                    lnew.set_visible(False)
                    
            if axmode.get_visible():
                if event.inaxes == axmode:
                    if not lmode.get_visible():
                        lmode.set_visible(True)
                else:
                    lmode.set_visible(False)
                    
            if axup.get_visible():
                if event.inaxes == axup:
                    if not lup.get_visible():
                        lup.set_visible(True)
                else:
                    lup.set_visible(False)
                    
            if axdown.get_visible():
                if event.inaxes == axdown:
                    if not ldown.get_visible():
                        ldown.set_visible(True)
                else:
                    ldown.set_visible(False)
                    
            if axgauss.get_visible():
                if event.inaxes == axgauss:
                    if not lgauss.get_visible():
                        lgauss.set_visible(True)
                else:
                    lgauss.set_visible(False)
                    
            if axstep.get_visible():
                if event.inaxes == axstep:
                    if not lstep.get_visible():
                        lstep.set_visible(True)
                else:
                    lstep.set_visible(False)
                    
            if axwall.get_visible():
                if event.inaxes == axwall:
                    if not lwall.get_visible():
                        lwall.set_visible(True)
                else:
                    lwall.set_visible(False)
        
        
        return True

    # If the mouse is inside the grid
    elif isinstance(event.xdata, float) and isinstance(event.ydata, float):

        linfo.set_visible(False)
        ldiff.set_visible(False)
        lmore.set_visible(False)
        lless.set_visible(False)
        lnew.set_visible(False)
        lmode.set_visible(False)
        lup.set_visible(False)
        ldown.set_visible(False)
        lgauss.set_visible(False)
        lstep.set_visible(False)
        lwall.set_visible(False)
        
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
            
            #only hover the remaining bricks
            if bricks(data)>0 and \
               (Y+1)>(bricks(data)+data_to_Vk(data)[pieces-1-X]):
                Y=bricks(data)+data_to_Vk(data)[pieces-1-X]-1
            #or if you are erasing
            elif bricks(data)<=0:
                if (Y+1)<=data_to_Vk(data)[pieces-1-X]:
                    Y=Y_hovered
                else:
                    Y=-1
            
            data[X][:Y+1]=1
            

            ball_the_data()
            ax.pcolormesh(np.transpose(data[::-1]), cmap=cmap(data), edgecolors=gridc, linewidths=gridw)
            event.canvas.draw()
            unball_the_data()
            
            data=data0
            
            
            return True
    


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

#language buttons 
#changes the language of the entire plot

def CAT(event):
    global trans, late, lated, language_reboot
    lated=late
    late=1
    language_reboot(event)
#    print(trans[late]['turtle'])

def ENG(event):
    global trans, late, lated, language_reboot
    lated=late
    late=0
    language_reboot(event)
#    print(trans[late]['turtle'])
    
def ESP(event):
    global trans, late, lated, language_reboot
    lated=late
    late=2
    language_reboot(event)
#    print(trans[late]['turtle'])
    
def language_reboot(event):
    global trans, late, detrans, lated,\
    lives, lives_txt, score_txt,\
    binfo, linfo, lmore, lless, bnew, lnew, bmode, lmode, lup, ldown,\
    demos_txt, lgauss, bstep, lstep, bwall, lwall, ldiff
    
    if lives==0:
        lives_txt.remove()    
        lives_txt = fig.text(0.815, 0.815, trans[late]['game over'])#, transform=ax.transAxes)
        score_txt.remove()    
        score_txt = fig.text(0.82, 0.765, trans[late]['score = %d']%(score),bbox={'facecolor':score_color, 'alpha':0.5, 'pad':5})#, transform=ax.transAxes)
    else:
        score_txt.remove()    
        score_txt = fig.text(0.82, 0.765, trans[late]['score = %d']%(score)*(int(lives/abs(lives))),bbox={'facecolor':score_color, 'alpha':0.5, 'pad':5})#, transform=ax.transAxes)
    
    
    binfo.label.set_text(trans[late][detrans[lated][binfo.label.get_text()]])
    linfo.set_text(trans[late][detrans[lated][linfo.get_text()]])
    
    ldiff.set_text(trans[late]['Change the amount of\npotential allowed'])
    
    lmore.set_text(trans[late]['Add a column'])
    lless.set_text(trans[late]['Remove a column'])
    
    bnew.label.set_text(trans[late]['New'])
    lnew.set_text(trans[late]['Go back to the\ninitial potential'])
    
    lmode.set_text(trans[late][detrans[lated][lmode.get_text()]])
    
    lup.set_text(trans[late]['   Go up 1\nenergy level'])
    ldown.set_text(trans[late][' Go down 1\nenergy level'])
    
    demos_txt.remove()
    demos_txt = fig.text(0.825, 0.385, trans[late]['demos'])#, transform=ax.transAxes)
    
    lgauss.set_text(trans[late]['  Draw our old friend\nthe Harmonic Oscillator'])
    
    bstep.label.set_text(trans[late]['Step'])
    lstep.set_text(trans[late]['Draw a high step'])
    
    bwall.label.set_text(trans[late]['Wall'])
    lwall.set_text(trans[late][' Draw a wall\non the middle'])
                        
#axbrick = plt.axes([0.820, 0.62, 0.05, 0.0375])
#bbrick = Button(axbrick, '', color='0.65', hovercolor='0.5')

axCAT = plt.axes([0.78, 0.94, 0.05, 0.0375])
bCAT = Button(axCAT, 'cat', color='0.65', hovercolor='0.5')
bCAT.on_clicked(CAT)

axENG = plt.axes([0.85, 0.94, 0.05, 0.0375])
bENG = Button(axENG, 'eng', color='0.65', hovercolor='0.5')
bENG.on_clicked(ENG)

axESP = plt.axes([0.92, 0.94, 0.05, 0.0375])
bESP = Button(axESP, 'esp', color='0.65', hovercolor='0.5')
bESP.on_clicked(ESP)

trans=[{'turtle':'turtle',      'game over':'game over',        'score = %d':'score = %d',  'Instructions: on':'Instructions: on',  'Instructions: off':'Instructions: off', 'Click to see\ninstructions\non balloons':'Click to see\ninstructions\non balloons',      'Click to stop\nseeing balloons':'Click to stop\nseeing balloons',      'Add a column':'Add a column',       'Remove a column':'Remove a\ncolumn',  'New':'New',  'Go back to the\ninitial potential':'Go back to the\ninitial potential',  'demos':'  demos',   '     Change to\nSURVIVAL mode':'     Change to\nSURVIVAL mode','Change to\nZEN mode':'Change to\nZEN mode',  '   Go up 1\nenergy level':'   Go up 1\nenergy level',   ' Go down 1\nenergy level':' Go down 1\nenergy level',  'Gauss':'Gauss', '  Draw our old friend\nthe Harmonic Oscillator':'  Draw our old friend\nthe Harmonic Oscillator',     'Step':'Step',   'Draw a high step':'Draw a high step',         'Wall':'Wall', ' Draw a wall\non the middle':' Draw a wall\non the middle', 'Change the amount of\npotential allowed':'Change the amount of\npotential allowed'},
       {'turtle':'tortuga',     'game over':'has perdut',       'score = %d':'punts = %d',  'Instructions: on':'Instruccions: sí',  'Instructions: off':'Instruccions:  no', 'Click to see\ninstructions\non balloons':'Clica per veure\ninstruccions\nen bafarades',  'Click to stop\nseeing balloons':'Clica per\ndeixar de veure\nbafarades','Add a column':'Afegeix\nuna columna','Remove a column':'Treu una\ncolumna', 'New':'Nou',  'Go back to the\ninitial potential':'Torna al\npotencial\ndel principi',   'demos':'exemples','     Change to\nSURVIVAL mode':'Canvia a mode\nsupervivència','Change to\nZEN mode':'Canvia a\nmode lliure','   Go up 1\nenergy level':"Puja 1 nivell\nd'energia",' Go down 1\nenergy level':"Baixa 1 nivell\nd'energia",'Gauss':'Gauss', '  Draw our old friend\nthe Harmonic Oscillator':"    Dibuixa al nostre\n          vell amic\nl'oscil·lador harmónic",'Step':'Esglaó', 'Draw a high step':'Dibuixa un\nesglaó alt',   'Wall':'Paret',' Draw a wall\non the middle':'Dibuixa una\nparet al mig', 'Change the amount of\npotential allowed':'Canvia la quantitat\nde potencial permès'},     
       {'turtle':'sapoconcha',  'game over':'fin de partida',   'score = %d':'puntos  %d',  'Instructions: on':'Instrucciones: si',  'Instructions: off':'Instrucciones: no', 'Click to see\ninstructions\non balloons':'Clica para ver\ninstructiones\nen bocadillos', 'Click to stop\nseeing balloons':'Clica para\ndejar de ver\nbocadillos', 'Add a column':'Añade una\ncolumna',  'Remove a column':'Quita una\ncolumna','New':'Nuevo','Go back to the\ninitial potential':'Vuelve al\npotencial\ndel principio', 'demos':'ejemplos','     Change to\nSURVIVAL mode':'Cambia a modo\nsupervivencia','Change to\nZEN mode':'Cambia a\nmodo libre', '   Go up 1\nenergy level':'Sube 1 nivel\nde enegia',' Go down 1\nenergy level':'Baja 1 nivel\nde energia', 'Gauss':'Gauss', '  Draw our old friend\nthe Harmonic Oscillator':'  Dibuja a nuestro\n   viejo amigo el\noscilador armónico','Step':'Escalón','Draw a high step':'  Dibuja un\nescalón alto','Wall':'Muro', ' Draw a wall\non the middle':'Dibuja\nun muro\nenmedio', 'Change the amount of\npotential allowed':'Cambia la cantidad de\npotencial permitido'}]

detrans=[{v: k for k, v in language.items()} for language in trans]

late=0
#print(trans[late]['turtle'])

    
#---------------------------------------------------------------------------

# info button
# labels everything with an explanation

def info(event):

    global trans, late, binfo, axinfo, linfo,\
    ball_value
    
    linfo.set_visible(False)  
    
    ball_value+=1

    if binfo.label.get_text()==trans[late]['Instructions: on']:

        binfo.label.set_text(trans[late]['Instructions: off'])
        axinfo.spines['top'].set_color((0,0,0.5))
        axinfo.spines['bottom'].set_color((0,0,0.5))
        axinfo.spines['right'].set_color((0,0,0.5))
        axinfo.spines['left'].set_color((0,0,0.5))
        
        linfo.set_text(trans[late]['Click to see\ninstructions\non balloons'])
        linfo.set_visible(False) 
        

    elif binfo.label.get_text()==trans[late]['Instructions: off']:

        binfo.label.set_text(trans[late]['Instructions: on'])
        axinfo.spines['top'].set_color((0,0,1))
        axinfo.spines['bottom'].set_color((0,0,1))
        axinfo.spines['right'].set_color((0,0,1))
        axinfo.spines['left'].set_color((0,0,1))
        
        linfo.set_text(trans[late]['Click to stop\nseeing balloons'])
        linfo.set_visible(False)    
    
    draw_wave()
    event.canvas.draw()

axinfo = plt.axes([0.78, 0.87, 0.19, 0.05])
binfo = Button(axinfo, trans[late]['Instructions: on'])
binfo.on_clicked(info)
axinfo.spines['top'].set_color((0,0,1))
axinfo.spines['bottom'].set_color((0,0,1))
axinfo.spines['right'].set_color((0,0,1))
axinfo.spines['left'].set_color((0,0,1))

el = Ellipse((2, -1), 0.5, 0.5)
axinfo.add_patch(el)

linfo = axinfo.annotate(trans[late]['Click to stop\nseeing balloons'], 
                        xy=(0,0.5), xytext=(-95,-10),
                        textcoords='offset points',
                        size=10, va='center',
                        bbox=dict(boxstyle='round', fc=(0.7, 0.7, 1), ec='none'),
                        arrowprops=dict(arrowstyle='wedge,tail_width=1.',
                                        fc=(0.7, 0.7, 1), ec='none',
                                        patchA=None,
                                        patchB=el,
                                        relpos=(0.2, 0.5)))

linfo.set_visible(False)

#---------------------------------------------------------------------------


# [ + ] button 
# adds an piece up to max_pieces

def more(event):
    global level, data, data_new, ax, Vk, pieces, txt, bricks, xk, kk, phi, max_pieces
    
    
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

        level=1
        draw_wave()
        event.canvas.draw()
        
        return True
        
    else: return True

axmore = plt.axes([0.825, 0.75, 0.05, 0.075])
bmore = Button(axmore, '$c+$')
bmore.on_clicked(more)
axmore.set_visible(False)

lmore = axmore.annotate(trans[late]['Add a column'], 
                        xy=(0,0.5), xytext=(-90,0),
                        textcoords='offset points',
                        size=10, va='center',
                        bbox=dict(boxstyle='round', fc=(0.7, 0.7, 1), ec='none'),
                        arrowprops=dict(arrowstyle='wedge,tail_width=1.',
                                        fc=(0.7, 0.7, 1), ec='none',
                                        patchA=None,
                                        patchB=el,
                                        relpos=(0.2, 0.5)))
lmore.set_visible(False)


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

axless = plt.axes([0.875, 0.75, 0.05, 0.075])
bless = Button(axless, '$c-$')
bless.on_clicked(less)
axless.set_visible(False)

lless = axless.annotate(trans[late]['Remove a column'], 
                        xy=(0,0.5), xytext=(-100,0),
                        textcoords='offset points',
                        size=10, va='center',
                        bbox=dict(boxstyle='round', fc=(0.7, 0.7, 1), ec='none'),
                        arrowprops=dict(arrowstyle='wedge,tail_width=1.',
                                        fc=(0.7, 0.7, 1), ec='none',
                                        patchA=None,
                                        patchB=el,
                                        relpos=(0.2, 0.5)))
lless.set_visible(False)
    
#---------------------------------------------------------------------------

# total lives and score

lives_txt = fig.text(0.935, 0.875, '<3 '*lives)#, transform=ax.transAxes)
score_txt = fig.text(0.885, 0.875, 'x %d'%(-max_value))#, transform=ax.transAxes)

#---------------------------------------------------------------------------

# new botton
# draws the squared infinite potential (nothing)

def new(event):

    global trans, late, level, data, data_new, score, cid_click, fig, score, lives, \
    ball_value, bmode, x_ball, y_ball, demos_txt
    
    data = np.copy(data_new)
    
    cid_click = fig.canvas.mpl_connect('button_press_event', ClickColor)

    axup.set_visible(True)
    bup.set_active(True)
    axdown.set_visible(True)
    bdown.set_active(True)
    axgauss.set_visible(True)
    bgauss.set_active(True)
    axstep.set_visible(True)
    bstep.set_active(True)
    axwall.set_visible(True)
    bwall.set_active(True)
    demos_txt.remove()
    demos_txt = fig.text(0.825, 0.385, trans[late]['demos'])#, transform=ax.transAxes)
    if bmode.label.get_text()=='SURVIVAL':
        score=0
        lives=3
        x_ball=0
        y_ball=9
    
    level=1
    draw_wave()
    event.canvas.draw()

axnew = plt.axes([0.825, 0.65, 0.1, 0.075])
bnew = Button(axnew, trans[late]['New'])
bnew.on_clicked(new)

lnew = axnew.annotate(trans[late]['Go back to the\ninitial potential'], 
                        xy=(0,0.5), xytext=(-95,0),
                        textcoords='offset points',
                        size=10, va='center',
                        bbox=dict(boxstyle='round', fc=(0.7, 0.7, 1), ec='none'),
                        arrowprops=dict(arrowstyle='wedge,tail_width=1.',
                                        fc=(0.7, 0.7, 1), ec='none',
                                        patchA=None,
                                        patchB=el,
                                        relpos=(0.2, 0.5)))
lnew.set_visible(False)

#---------------------------------------------------------------------------

# mode botton
# changes from zen mode to survival mode and vs

def mode(event):

    global trans, late, level, data, data_new, score, \
    cid_click, fig, score, lives, ball_value, \
    bmode, axmode, lmode, axmore, axless, pieces, demos_txt

    lmode.set_visible(False)

    data = np.copy(data_new)   

    if bmode.label.get_text()=='SURVIVAL':
        bmode.label.set_text('ZEN')
        axmode.spines['top'].set_color((0,1,0))
        axmode.spines['bottom'].set_color((0,1,0))
        axmode.spines['right'].set_color((0,1,0))
        axmode.spines['left'].set_color((0,1,0))
#        print(bmode.label.get_text())
        cid_click = fig.canvas.mpl_connect('button_press_event', ClickColor)
        lives=-1
        
        axmore.set_visible(True)
        bmore.set_active(True)
        axless.set_visible(True)
        bless.set_active(True)
        axup.set_visible(True)
        bup.set_active(True)
        axdown.set_visible(True)
        bdown.set_active(True)
        axgauss.set_visible(True)
        bgauss.set_active(True)
        axstep.set_visible(True)
        bstep.set_active(True)
        axwall.set_visible(True)
        bwall.set_active(True)
        
        
        demos_txt.remove()
        demos_txt = fig.text(0.825, 0.385, trans[late]['demos'])#, transform=ax.transAxes)
      
    
        lmode.set_text(trans[late]['     Change to\nSURVIVAL mode'])
        lmode.set_visible(False)
        
    elif bmode.label.get_text()=='ZEN':
        if pieces!=7:
            pieces=6
            more(event)
        bmode.label.set_text('SURVIVAL')
        axmode.spines['top'].set_color((1,0,0))
        axmode.spines['bottom'].set_color((1,0,0))
        axmode.spines['right'].set_color((1,0,0))
        axmode.spines['left'].set_color((1,0,0))
#        print(bmode.label.get_text())
        cid_click = fig.canvas.mpl_connect('button_press_event', ClickColor)
        lives=3
        score=0
        axmore.set_visible(False)
        bmore.set_active(False)
        axless.set_visible(False)
        bless.set_active(False)
        
        lmode.set_text(trans[late]['Change to\nZEN mode'])
        lmode.set_visible(False)
    
    level=1
    draw_wave()
    event.canvas.draw()

axmode = plt.axes([0.815, 0.55, 0.12, 0.075])
bmode = Button(axmode, 'SURVIVAL')
bmode.on_clicked(mode)
axmode.spines['top'].set_color((1,0,0))
axmode.spines['bottom'].set_color((1,0,0))
axmode.spines['right'].set_color((1,0,0))
axmode.spines['left'].set_color((1,0,0))

lmode = axmode.annotate(trans[late]['Change to\nZEN mode'], 
                        xy=(0,0.5), xytext=(-95,0),
                        textcoords='offset points',
                        size=10, va='center',
                        bbox=dict(boxstyle='round', fc=(0.7, 0.7, 1), ec='none'),
                        arrowprops=dict(arrowstyle='wedge,tail_width=1.',
                                        fc=(0.7, 0.7, 1), ec='none',
                                        patchA=None,
                                        patchB=el,
                                        relpos=(0.2, 0.5)))
lmode.set_visible(False)

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
        draw_wave()
        event.canvas.draw()
        
        return True
        
    else: return True

axup = plt.axes([0.825, 0.45, 0.05, 0.075])
bup = Button(axup, '$E+$')
bup.on_clicked(up)

lup = axup.annotate(trans[late]['   Go up 1\nenergy level'], 
                        xy=(0,0.5), xytext=(-80,0),
                        textcoords='offset points',
                        size=10, va='center',
                        bbox=dict(boxstyle='round', fc=(0.7, 0.7, 1), ec='none'),
                        arrowprops=dict(arrowstyle='wedge,tail_width=1.',
                                        fc=(0.7, 0.7, 1), ec='none',
                                        patchA=None,
                                        patchB=el,
                                        relpos=(0.2, 0.5)))
lup.set_visible(False)

#---------------------------------------------------------------------------

# [ ↓ ] button
# changes from any level down to the first one

def down(event):
    global level, N_root

    if level>1 and level<=N_root:  
        level-=1
#        print(level)
        draw_wave()
        event.canvas.draw()
        
        return True
        
    else: return True

axdown = plt.axes([0.875, 0.45, 0.05, 0.075])
bdown = Button(axdown, '$E-$')
bdown.on_clicked(down)

ldown = axdown.annotate(trans[late][' Go down 1\nenergy level'], 
                        xy=(0,0.5), xytext=(-80,0),
                        textcoords='offset points',
                        size=10, va='center',
                        bbox=dict(boxstyle='round', fc=(0.7, 0.7, 1), ec='none'),
                        arrowprops=dict(arrowstyle='wedge,tail_width=1.',
                                        fc=(0.7, 0.7, 1), ec='none',
                                        patchA=None,
                                        patchB=el,
                                        relpos=(0.2, 0.5)))
ldown.set_visible(False)

#---------------------------------------------------------------------------

# demo buttons
# demos are drawn by hand for pieces=1..7 only

demos_txt = fig.text(0.825, 0.385, trans[late]['demos'])#, transform=ax.transAxes)

def Vk_to_data(Vk):
    
    global data_new
    
    translated_data = np.copy(data_new)
    
    for X in range(len(Vk)):
        Y=Vk[X]-1
        translated_data[X][:Y+1]=2
        translated_data[X][Y+1:]=0
    
    return translated_data


def gauss(event):
    global level, data, pieces
    

    Vk=[[0],
        [0,0],[5,1,5],
        [5,1,1,5],[8,2,0,2,8],
        [6,1,0,0,1,6],[9,4,1,0,1,4,9]]

    if pieces<=7:
        data = Vk_to_data(Vk[pieces-1])
        level=1
        draw_wave()
        event.canvas.draw()
    else:
        return True

axgauss = plt.axes([0.825, 0.285, 0.1, 0.075])
bgauss = Button(axgauss, 'Gauss')
bgauss.on_clicked(gauss)

lgauss = axgauss.annotate(trans[late]['  Draw our old friend\nthe Harmonic Oscillator'], 
                        xy=(0,0.5), xytext=(-130,0),
                        textcoords='offset points',
                        size=10, va='center',
                        bbox=dict(boxstyle='round', fc=(0.7, 0.7, 1), ec='none'),
                        arrowprops=dict(arrowstyle='wedge,tail_width=1.',
                                        fc=(0.7, 0.7, 1), ec='none',
                                        patchA=None,
                                        patchB=el,
                                        relpos=(0.2, 0.5)))
lgauss.set_visible(False)


def step(event):
    global level, data, pieces

    Vk=[[0],
        [0,9],[0,9,9],
        [0,0,9,9],[0,0,9,9,9],
        [0,0,0,9,9,9],[0,0,0,9,9,9,9]]

    if pieces<=7:
        data = Vk_to_data(Vk[pieces-1])
        level=1
        draw_wave()
        event.canvas.draw()
    else:
        return True

axstep = plt.axes([0.825, 0.185, 0.1, 0.075])
bstep = Button(axstep, trans[late]['Step'])
bstep.on_clicked(step)

lstep = axstep.annotate(trans[late]['Draw a high step'], 
                        xy=(0,0.5), xytext=(-100,0),
                        textcoords='offset points',
                        size=10, va='center',
                        bbox=dict(boxstyle='round', fc=(0.7, 0.7, 1), ec='none'),
                        arrowprops=dict(arrowstyle='wedge,tail_width=1.',
                                        fc=(0.7, 0.7, 1), ec='none',
                                        patchA=None,
                                        patchB=el,
                                        relpos=(0.2, 0.5)))
lstep.set_visible(False)


def wall(event):
    global data, pieces, level

    Vk=[[0],[0,0],
        [0,10,0],
        [0,10,10,0],
        [0,0,10,0,0],
        [0,0,10,10,0,0],
        [0,0,0,10,0,0,0]]

    if pieces<=7:
        data = Vk_to_data(Vk[pieces-1])
        level=1
        draw_wave()
        event.canvas.draw()
    else:
        return True

axwall = plt.axes([0.825, 0.085, 0.1, 0.075])
bwall = Button(axwall, trans[late]['Wall'])
bwall.on_clicked(wall)

lwall = axwall.annotate(trans[late][' Draw a wall\non the middle'], 
                        xy=(0,0.5), xytext=(-80,0),
                        textcoords='offset points',
                        size=10, va='center',
                        bbox=dict(boxstyle='round', fc=(0.7, 0.7, 1), ec='none'),
                        arrowprops=dict(arrowstyle='wedge,tail_width=1.',
                                        fc=(0.7, 0.7, 1), ec='none',
                                        patchA=None,
                                        patchB=el,
                                        relpos=(0.2, 0.5)))
lwall.set_visible(False)

#---------------------------------------------------------------------------

# difficulty button

# remaining bricks
# Counts how many bricks you have.
# at 6*pieces people start to get creative.

def bricks(data):
    global pieces,difficulty,bmode
    
    if bmode.label.get_text()=='SURVIVAL':
        bricks0 = difficulty*pieces
    else:
        bricks0 = 10*pieces
    
    return bricks0-np.sum(data_to_Vk(data))

#axbrick = plt.axes([0.820, 0.62, 0.05, 0.0375])
#bbrick = Button(axbrick, '', color='0.65', hovercolor='0.5')
#
#txt = fig.text(0.885, 0.625, 'x %d'%(int(bricks(data))), transform=ax.transAxes)

def diff(event):

    global trans, late, new, \
    cid_click, fig, \
    bdiff, axdiff, ldiff, difficulty, score_color, ball_color

    ldiff.set_visible(False) 

    if bdiff.label.get_text()=='EASY':
        difficulty=5
        bdiff.label.set_text('FAIR')
        score_color=(1,1,0)
        ball_color=(1,1,0)
        axdiff.spines['top'].set_color((1,1,0))
        axdiff.spines['bottom'].set_color((1,1,0))
        axdiff.spines['right'].set_color((1,1,0))
        axdiff.spines['left'].set_color((1,1,0))
        cid_click = fig.canvas.mpl_connect('button_press_event', ClickColor)
        
        
    elif bdiff.label.get_text()=='FAIR' or bdiff.label.get_text()=='FAIR.':
        difficulty=4
        bdiff.label.set_text('HARD')
        score_color=(1,0,0)
        ball_color=(1,0,0)
        axdiff.spines['top'].set_color((1,0,0))
        axdiff.spines['bottom'].set_color((1,0,0))
        axdiff.spines['right'].set_color((1,0,0))
        axdiff.spines['left'].set_color((1,0,0))
        cid_click = fig.canvas.mpl_connect('button_press_event', ClickColor)
        
        
    elif bdiff.label.get_text()=='HARD':
        difficulty=10
        score_color=(0,1,0)
        ball_color=(0,1,0)
        bdiff.label.set_text('EASY')
        axdiff.spines['top'].set_color((0,1,0))
        axdiff.spines['bottom'].set_color((0,1,0))
        axdiff.spines['right'].set_color((0,1,0))
        axdiff.spines['left'].set_color((0,1,0))
        cid_click = fig.canvas.mpl_connect('button_press_event', ClickColor)
    
    draw_wave()
    event.canvas.draw()
    new(event)

axdiff = plt.axes([0.02, 0.92, 0.08, 0.05])
bdiff = Button(axdiff, 'EASY')
bdiff.on_clicked(diff)
axdiff.spines['top'].set_color((0,1,0))
axdiff.spines['bottom'].set_color((0,1,0))
axdiff.spines['right'].set_color((0,1,0))
axdiff.spines['left'].set_color((0,1,0))

ldiff = axdiff.annotate(trans[late]['Change the amount of\npotential allowed'], 
                        xy=(1,0.5), xytext=(+35,-20),
                        textcoords='offset points',
                        size=10, va='center',
                        bbox=dict(boxstyle='round', fc=(0.7, 0.7, 1), ec='none'),
                        arrowprops=dict(arrowstyle='wedge,tail_width=1.',
                                        fc=(0.7, 0.7, 1), ec='none',
                                        patchA=None,
                                        patchB=el,
                                        relpos=(0.2, 0.5)))
ldiff.set_visible(False)

#---------------------------------------------------------------------------
# SCORE 
# How survival mode works
# The green square is calles ball
#---------------------------------------------------------------------------

def play_ball(touched):
    global x_ball, y_ball, ball_value, max_value, score, lives,\
        bmode, axup,axdown,axgauss,axstep,axwall,demos_txt,\
        no_bricks,bdiff
    
    if bmode.label.get_text()=='SURVIVAL':
        if x_ball==0 and y_ball==9:
            new_ball(touched)
            ball_value+=1
        if touched[len(touched)-1-x_ball]>=y_ball:
#            print('%d + %d = %d'%(score,ball_value+1,score+ball_value+1))
            score+=ball_value
            new_ball(touched)
        else:
            try:
                ball_value
            
            except NameError: #The game just started
                new_ball(touched)
                
            else:
                ball_value-=1
                if no_bricks:
                    print('stupid')
                    ball_value+=1
                    no_bricks=False
                if ball_value==0:
                    lives-=1
                    if lives==0:
                        axup.set_visible(False)
                        bup.set_active(False)
                        axdown.set_visible(False)
                        bdown.set_active(False)
                        axgauss.set_visible(False)
                        bgauss.set_active(False)
                        axstep.set_visible(False)
                        bstep.set_active(False)
                        axwall.set_visible(False)
                        bwall.set_active(False)
                        demos_txt.remove()
                        demos_txt = fig.text(0.838, 0.425, '')#, transform=ax.transAxes)
    #                    print('game over')
                    else:
    #                    print('bum')
                        new_ball(touched)
        

      
    elif bmode.label.get_text()=='ZEN':
        x_ball=0
        y_ball=9
        ball_value=0

    

def new_ball(touched):
    global x_ball, y_ball, ball_value, bdiff
    
    x_ball  = random.randint(0,len(touched)-1)
    while touched[len(touched)-1-x_ball]>7:
        x_ball  = random.randint(0,len(touched)-1)
    y_ball  = random.randint(touched[len(touched)-1-x_ball]+1,9)
    ball_value = y_ball
    
    if x_ball==0 or x_ball==len(touched)-1:
        if y_ball==9:
            new_ball(touched)
            
    if y_ball==1 or y_ball==0:
        new_ball(touched)

    if bdiff.label.get_text()=='FAIR':
        if x_ball==0 and y_ball==8:
            new_ball(touched)
        if y_ball>8:
            new_ball(touched)
        
    if bdiff.label.get_text()=='HARD':
        if x_ball==0 and y_ball==8:
            new_ball(touched)
        if y_ball>7:
            new_ball(touched)
        
            
value_txt = fig.text(0.885, 0.925, '%d'%(max_value))#, transform=ax.transAxes)

def ball_the_data():
    global x_ball,y_ball,data,pre_ball
    pre_ball=data[x_ball,y_ball]
    if bmode.label.get_text()=='SURVIVAL':
        data[x_ball,y_ball]=3
    return data

def unball_the_data():
    global x_ball,y_ball,data,pre_ball
    data[x_ball,y_ball]=pre_ball
    return data

#---------------------------------------------------------------------------

# end
fig.text(0.865, 0.04, 'draw my pot', fontsize='smaller')
fig.text(0.775, 0.01, 'rahensilva@gmail.com', fontsize='smaller')

draw_wave()

plt.show()





















