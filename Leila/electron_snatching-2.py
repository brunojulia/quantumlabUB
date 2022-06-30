from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#Paràmetres
nx=500
lmax=50
lmin=-10
lx=lmax-lmin
dx=lx/(nx-1)
dt=0.5

# Constants
hbarra=1
m=1
r=(hbarra**2)/(2*m*(dx**2))
mu_phi=0.0
sigma_phi=1

# Pozo
omega=0.1
y1=3
desplazamiento=0.0
ancho=6.0

xx=np.linspace(lmin,lmax,nx)

def phi0(nx,xx):
	phi0=1j*np.zeros(nx)
	for i in range(0,nx):
		phi0[i]=((1.0/(sigma_phi*np.sqrt(2.0*np.pi)))*(np.exp(-(1/2)*((xx[i]-mu_phi)/(sigma_phi))**2)))
	return phi0

# Matriu H -> minicodi 3-hamiltonia
def hamiltonia(nx,r):
	H=1j*np.zeros((nx,nx))

	for i in range(0,nx):
		H[i,i]=(2*r)           # diagonal central
		if i != (nx-1):
			H[i,i+1]=(-1)*r	   # diagonal superior
			H[i+1,i]=(-1)*r    # diagonal inferior
	return H

# Matriu B -> minicodi 4-matriuB
def matriuB(nx,dt,H,hbarra):
	matriu_B=1j*np.zeros((nx,nx))

	for i in range(0,nx):
		matriu_B[i,i]=1-(1j*dt/(2*hbarra))*H[i,i]           # diagonal central
		if i != (nx-1):
			matriu_B[i,i+1]=1-(1j*dt/(2*hbarra))*H[i,i+1]	# diagonal superior
			matriu_B[i+1,i]=1-(1j*dt/(2*hbarra))*H[i+1,i]	# diagonal inferior
	return matriu_B

# Vectors abc -> minicodi 5-abc
def abc(nx,dt,H,hbarra):
	a=1j*np.zeros(nx)    # Diagonal Central
	b=1j*np.zeros(nx)    # Diagonal Inferior
	c=1j*np.zeros(nx)    # Diagonal Superior

	for i in range(0,nx):
		b[i]=1+(1j*dt/(2*hbarra))*H[i,i]
		if i != 0:
			a[i]=1+(1j*dt/(2*hbarra))*H[i,i-1]
		if i != (nx-1):
			c[i]=1+(1j*dt/(2*hbarra))*H[i,i+1]

	return a,b,c

# funció tridiagonal -> minicodi 6-tridiag
def tridiag(a,b,c,d):

#FUNCIÓ TRIDIAGONAL
#Introduïm els vectors de les 3 diagonal i el vector d

	n=len(a)   #Nombre de files (equacions) 

	#Creamos 2 vectores para los nuevos coeficientes:
	cp=1j*np.zeros(n)
	dp=1j*np.zeros(n)

	#Modificamos los coeficientes de la primera fila
	# i los guardamos en los nuevos vectores
	cp[0] = c[0]/b[0] 
	dp[0] = d[0]/b[0]

	for i in range(1,n):
		denom=(b[i]-a[i]*cp[i-1])
		cp[i]=c[i]/denom
		dp[i]=(d[i]-a[i]*dp[i-1])/denom

	phi=1j*np.zeros(n)
	phi[n-1]=dp[n-1]

	for j in range(1,n):
		i=(n-1)-j
		phi[i]=dp[i]-cp[i]*phi[i+1]

	return phi

def pozo(nx,xx,ancho,despl):
	pozo=1j*np.zeros(nx)
	xe=np.abs(np.sqrt(2*y1/(ancho*m*omega**2)))
	for i in range(0,nx):
		x=xx[i]
		if x <= -xe+desplazamiento:
			pozo[i]=y1
		elif x >= xe+desplazamiento:
			pozo[i]=y1
		else:
			pozo[i]=ancho*(m/2.0)*(omega*(x-desplazamiento))**2
	return pozo

def modul(nx,f):
	phi_abs = np.zeros(nx)
	for i in range(0,nx):
		phi_abs=(abs(f))
	return phi_abs

# PARÁMETRES EVOLUCIÓ
H=hamiltonia(nx,r)
matriu_B=matriuB(nx,dt,H,hbarra)
a,b,c=abc(nx,dt,H,hbarra)

# ---------------------- VENTANA ------------------------

# Creamos la ventana
ventana=Tk()
ventana.title("Electron-snatching")   # Título de la ventana
ventana.geometry("900x500")           # Dimension ventana
ventana.config(bg="indigo")


# Establecemos Filas y Columnas de la ventana
ventana.columnconfigure(index=1,weight=1)
ventana.rowconfigure(index=0,weight=8)

ventana.columnconfigure(index=0,weight=1)
ventana.rowconfigure(index=1,weight=1)

ventana.columnconfigure(index=1,weight=1)
ventana.rowconfigure(index=1,weight=1)

ventana.columnconfigure(index=2,weight=1)
ventana.rowconfigure(index=1,weight=1)

# Cramos los Frames y los colocamos en las celdas de la ventana
# Logo
frame1=Frame(ventana, bg="indigo")
frame1.grid(columnspan=3,row=0,sticky='snew')

frame1.columnconfigure(index=0, weight=1)
frame1.rowconfigure(index=0,weight=1)

frame1.columnconfigure(index=0, weight=1)
frame1.rowconfigure(index=1,weight=1)

frame1.columnconfigure(index=0, weight=1)
frame1.rowconfigure(index=2,weight=40)


# Cronometro
frame2=Frame(ventana, bg="grey")

frame2.columnconfigure(index=0, weight=1)
frame2.rowconfigure(index=0,weight=1)


# Etiquetas con valores
frame3=Frame(ventana, bg="grey")

#Sigma
frame3.columnconfigure(index=0, weight=1)
frame3.rowconfigure(index=0,weight=1)

#Mu
frame3.columnconfigure(index=0, weight=1)
frame3.rowconfigure(index=1,weight=1)

# SLIDERS
frame4=Frame(ventana, bg="grey")

#Sigma
frame4.columnconfigure(index=0, weight=1)
frame4.rowconfigure(index=0,weight=1)

#Mu
frame4.columnconfigure(index=0, weight=1)
frame4.rowconfigure(index=1,weight=1)


frame5=Frame(ventana, bg='indigo')


# Insertamos el logo
canvas1=Canvas(frame1, bd=0, highlightthickness=0, bg="indigo")
canvas1.grid(column=0,row=0)
logo=PhotoImage(file="logo.png")
canvas1.create_image(140, 50, image=logo, anchor="n")

# Insertamos el título
canvas2=Canvas(frame1, bd=0, highlightthickness=0, bg="indigo")
canvas2.grid(column=0,row=1)
canvas2.create_text(150, 50, text="ELECTRON\nSNATCHING", fill="white", font=('Chalkduster 40'))


# -----------------------START ----------------------

def start_boton():
	cuenta_atras.grid(row=1,column=0)

	start.grid_forget()
	canvas1.delete('all')
	canvas2.delete('all')
	countdown()

# --------------------- COUNTDOWN ---------------------

t=5
def countdown():
	global t
	if t>0:
		cuenta_atras.config(text=t)
		t=t-1
		cuenta_atras.after(1000,countdown)
	elif t==0:
		cuenta_atras.config(text='Go!')
		cuenta_atras.after(1000, inicio)

# ---------------------- INICIO -------------------------

def inicio():
	cuenta_atras.grid_remove()
	frame1.grid_remove()

	frame2.grid(column=0,row=1,sticky='snew')
	frame3.grid(column=1,row=1,sticky='snew')
	frame4.grid(column=2,row=1,sticky='snew')
	frame5.grid(columnspan=3,row=0,sticky='snew')

	crono()
	grafic_inicial()


# --------------------- CRONÓMETRO ---------------------
minuto=0
segundo=0
mili=0
def crono():
	global minuto, segundo, mili
	mili=mili+1
	if mili == 999:
		mili=0
		segundo=segundo+1
		if segundo == 59:
			segundo=0
			minuto=minuto+1

	m="{:0>2d}".format(minuto)
	s="{:0>2d}".format(segundo)
	ms="{:0>3d}".format(mili)
	cronometro.grid(row=0,column=0)
	cronometro.config(text=str(m) + ":" + str(s) + "." + str(ms))
	cronometro.after(1,crono)

# ----------------------- GRÁFICO ---------------------

fig=Figure(figsize=(10,5), dpi=100)

canvas=FigureCanvasTkAgg(fig, master=frame5) # Crear area de dibujo
canvas.get_tk_widget().grid(column=0,row=0, padx=80, pady=10)
ax=fig.add_subplot(111)

ax.set_xlim(-10, 50)
ax.set_ylim(-1, 5)

def grafic_inicial():
	global phi1
	# Pozo
	pou=pozo(nx,xx,ancho,desplazamiento)
	pou_abs=modul(nx,pou)

	# Funcion phi0 inicial
	phi1=phi0(nx,xx)
	phi1_abs=modul(nx,phi1)

	graf=pou_abs+phi1_abs

	ax.plot(xx,pou_abs)
	ax.plot(xx,graf)
	ax.fill_between(xx,pou_abs,graf, color='orange')
	canvas.draw()

	print('aaaaaaaaaaah')

	valor_ancho.config(text=str(ancho))
	valor_ancho.after(500, update)

def valor1(val1):
	global ancho, desplazamiento
	valor_ancho.config(text=val1)
	ancho=float(val1)

def valor2(val2):
	global ancho, desplazamiento
	valor_desplazamiento.config(text=val2)
	desplazamiento=float(val2)


def update():
	global ancho,desplazamiento,phi1
	ax.clear()

	# POZO
	pou=pozo(nx,xx,ancho,desplazamiento)
	pou_abs=modul(nx,pou)

	# PHI
	# part dreta de l'equació d=B*phi
	d=np.dot(matriu_B,phi1)

	# Apliquem condicions de contorn
	d[0]=0
	d[nx-2]=0

	# Calculem
	phi2=tridiag(a,b,c,d)

	# Calculem el mòdul i la norma
	phi2_abs=modul(nx,phi2)
	graf2=phi2_abs+pou_abs

	ax.set_xlim(-10, 50)
	ax.set_ylim(-1, 5)
	ax.plot(xx,pou_abs)
	ax.plot(xx,graf2)
	ax.fill_between(xx,pou_abs,graf2, color='orange')
	canvas.draw()

	phi1=phi2
	valor_ancho.config(text=str(ancho))
	valor_ancho.after(500, update)


# ------------------------- WIDGETS -----------------------

# Creamos el boton de start
start = Button(frame1, text='START', bd=3, fg='purple4', relief = "raised",
				font=('Arial', 20, 'bold'), width=20, height=3,
				command=start_boton)
start.configure(bg='blue')
start.grid(column=0, row=2, sticky='n')

cuenta_atras=Label(frame1 ,bg='indigo', fg='white', font=('Arial', 100, 'bold'))

cronometro=Label(frame2, bg='grey', fg='white', font=('Arial', 30, 'bold'))

valor_ancho = Label(frame3, text=ancho, width = 15)
valor_ancho.grid(column=0, row=0, pady =5)

valor_desplazamiento = Label(frame3, text=desplazamiento, width = 15)
valor_desplazamiento.grid(column=0, row=1, pady =5)

slider_ancho = Scale(frame4, from_ =6, to = 20, resolution = 0.01,
				showvalue=NO, 
			  orient='horizontal', length=300, command=valor1)
slider_ancho.grid(column=0, row=0)


slider_desplazamiento = Scale(frame4, from_ =1, to = 40, resolution = 0.01,
			  showvalue=NO, orient='horizontal', length=300, command=valor2)
slider_desplazamiento.grid(column=0, row=1)


 

ventana.mainloop()




