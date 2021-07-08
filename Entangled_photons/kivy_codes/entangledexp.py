#En aquest document aniran totes les coses relacionades amb el càlcul de l'experiment per després importar-lo en el "entangled".
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

class entangledEXP(object):
	#aqui posem totes les propietats del entangledEXP que escriurem com self. el que sigui
	def __init__(self,n = 10000, theta_l=math.pi/4, phi=26*math.pi/180,photons=10000, b1=5*math.pi/36, b2=math.pi/2):
		self.n = n
		self.tl = theta_l
		self.phi = phi
		self.photons = photons
		self.b1 = b1
		self.b2 = b2

	def addphotons(self, n=None):
		changed=False
		if n is not None:
			changed=changed or int(self.n)!=int(n) #changed agafa el valor true o false si changed=True o si n no és self.n
			self.n = n
			return changed

# ----------------------------EXPERIMENT QUÀNTIC-------------------------------
	def verticals(self,a1,b1,):
		# prob VV
		verticals = math.sin(a1) ** 2 * math.sin(b1) ** 2 * math.cos(self.tl) ** 2 + math.cos(
			a1) ** 2 * math.cos(b1) ** 2 * math.sin(self.tl) ** 2 + (1 / 4) * math.sin(2 * a1) * math.sin(
			2 * b1) * math.sin(2 * self.tl) * math.cos(self.phi)
		return (verticals)

	def crossed(self,a1,b1):
		# prob HV o VH
		Pcrossed = math.sin(a1) ** 2 * math.cos(b1) ** 2 * math.cos(self.tl) ** 2 - (1 / 4) * math.sin(
			2 * a1) * math.sin(2 * b1) * \
				   math.sin(2 * self.tl) * math.cos(self.phi) + math.cos(a1) ** 2 * math.sin(b1) ** 2 * math.sin(
			self.tl) ** 2
		return (Pcrossed)

	def horizontals(self,a1,b1):
		Phh = math.cos(a1) ** 2 * math.cos(b1) ** 2 * math.cos(self.tl) ** 2 + math.sin(a1) ** 2 * math.sin(
			b1) ** 2 * math.sin(self.tl) ** 2 + \
			  (1 / 4) * math.sin(2 * a1) * math.sin(2 * b1) * math.sin(2 * self.tl) * math.cos(self.phi)
		return (Phh)

	def expqua(self, alpha, beta):
		# creem les llistes dels angles que posarem als polaritzadors per poder calcular S
		# fem que els angles estiguin entre -pi i pi
		pol1 = []
		pol2 = []
		for i in range(0, 4):
			if (alpha + i * math.pi / 4) > math.pi:
				pol1.append(alpha + i * math.pi / 4 - 2 * math.pi)
			else:
				pol1.append(alpha + i * math.pi / 4)
		for i in range(0, 4):
			if beta + i * math.pi / 4 > math.pi:
				pol2.append(beta + i * math.pi / 4 - 2 * math.pi)
			else:
				pol2.append(beta + i * math.pi / 4)
		table1 = []
		for a1 in pol1:
			for b1 in pol2:
				VV = int(self.verticals(a1, b1) * self.photons)
				VH = int(self.crossed(a1, b1) * self.photons)
				HV = int(self.crossed(b1, a1) * self.photons)  # simplement canviar alpha per beta i viceversa

				resultats = [a1, b1, VH, HV, VV]
				table1.append(resultats)
				resultats = []

		return (table1)


	def scalc(self, alpha, beta):
		table1=self.expqua(alpha, beta)
		Elist = []
		# Nc=llista de coincidències, agafa la última columna de la taula
		Nc = []
		for result in table1:
			Nc.append(result[4])
		# posicions en la llista de coincidències de E(alpha,beta). Ho faig així perquè segueixen un patró.
		posab = [0, 1, 4, 5]
		for i in posab:
			# calcula la E segons la posició de E(alpha,beta) de la llista de coincidències
			E = (Nc[i] + Nc[i + 10] - Nc[i + 2] - Nc[i + 8]) / (Nc[i] + Nc[i + 10] + Nc[i + 2] + Nc[i + 8])
			Elist.append(E)
		# Agafem els angles a=-45, a'=0 i b=22.5, b'=-22.5
		# print(Elist)
		S = Elist[0] - Elist[1] + Elist[2] + Elist[3]

		return (S)

	scalcvec=np.vectorize(scalc)

	def sigma(self, alpha, beta):
		# Nc=llista de coincidències, agafa la última columna de la taula
		table1=self.expqua(alpha, beta)
		Nc = []
		for result in table1:
			Nc.append(result[4])
		# posicions en la llista de coincidències de E(alpha,beta).
		posab = [0, 1, 4, 5]
		##################
		# Càlcul de la desviació estàndard
		##################
		suma = []
		for i in posab:
			# calcula la derivada de E segons la posició de E(alpha,beta) de la llista de coincidències
			# derivades
			dEi1 = 2 * (Nc[i + 2] + Nc[i + 8]) / (Nc[i] + Nc[i + 10] + Nc[i + 2] + Nc[i + 8]) ** 2
			dEi2 = -2 * (Nc[i] + Nc[i + 10]) / (Nc[i] + Nc[i + 10] + Nc[i + 2] + Nc[i + 8]) ** 2

			suma.append(dEi1 ** 2 * (Nc[i] + Nc[i + 10]) + dEi2 ** 2 * (Nc[i + 2] + Nc[i + 8]))

		sigma = math.sqrt(sum(suma))
		return(sigma)


	def sweepS(self):
		res = []
		# we take the lowest of b1 and b2
		angle = 0
		alphalist = np.linspace(0, 2 * np.pi, 200)
		if self.b1 > self.b2:
			angle1 = self.b2
			angle2 = self.b1
		else:
			angle1 = self.b1
			angle2 = self.b2
		# WARNING: max beta linspace points ~ 50
		print(angle1,angle2)
		betalist = np.linspace(angle1, angle2, 100)

		# create 2d x,y grid (both X and Y will be 2d)
		#X, Y = np.meshgrid(alphalist, betalist, sparse=True)
		# repeat Z to make it a 2d grid

		#Z = self.scalcvec(self,X, Y)

		coords=(alphalist, betalist)
		return(coords)
		# mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
		# mappable.set_array(Z)
		# fig = plt.figure()
		# ax = Axes3D(fig)
		# ax.plot_surface(X, Y, Z, cmap=mappable.cmap, linewidth=0.01)
		#
		# ax.set_xlabel('Alpha (rad)')
		# ax.set_ylabel('Beta (rad)')
		# ax.set_zlabel('S')
		#
		# cbar = fig.colorbar(mappable,shrink=0.5)
		# cbar.set_label('S', rotation=0)
		# plt.savefig('test.png')
		# plt.show()
