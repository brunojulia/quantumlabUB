# En aquest document aniran totes les coses relacionades amb el càlcul de l'experiment per després importar-lo en el "entangled".
import math
import numpy as np
import random
from kivy.properties import ListProperty


class entangledEXP(object):
    # aqui posem totes les propietats del entangledEXP que escriurem com self. el que sigui
    def __init__(self, n=10000, theta_l=math.pi / 4, phi=26 * math.pi / 180, photons=10000, b1=5 * math.pi / 36,
                 b2=math.pi / 2):
        self.n = n
        self.tl = theta_l
        self.phi = phi
        self.photons = photons
        self.b1 = b1
        self.b2 = b2

    def addphotons(self, n=None):
        changed = False
        if n is not None:
            changed = changed or int(self.n) != int(
                n)  # changed agafa el valor true o false si changed=True o si n no és self.n
            self.n = n
            return changed

    # ----------------------------EXPERIMENT QUÀNTIC-------------------------------
    def verticals(self, a1, b1, ):
        # prob VV
        verticals = math.sin(a1) ** 2 * math.sin(b1) ** 2 * math.cos(self.tl) ** 2 + math.cos(
            a1) ** 2 * math.cos(b1) ** 2 * math.sin(self.tl) ** 2 + (1 / 4) * math.sin(2 * a1) * math.sin(
            2 * b1) * math.sin(2 * self.tl) * math.cos(self.phi)
        return verticals

    def crossed(self, a1, b1):
        # prob HV o VH
        Pcrossed = math.sin(a1) ** 2 * math.cos(b1) ** 2 * math.cos(self.tl) ** 2 - (1 / 4) * math.sin(
            2 * a1) * math.sin(2 * b1) * \
                   math.sin(2 * self.tl) * math.cos(self.phi) + math.cos(a1) ** 2 * math.sin(b1) ** 2 * math.sin(
            self.tl) ** 2
        return Pcrossed

    def horizontals(self, a1, b1):
        Phh = math.cos(a1) ** 2 * math.cos(b1) ** 2 * math.cos(self.tl) ** 2 + math.sin(a1) ** 2 * math.sin(
            b1) ** 2 * math.sin(self.tl) ** 2 + \
              (1 / 4) * math.sin(2 * a1) * math.sin(2 * b1) * math.sin(2 * self.tl) * math.cos(self.phi)
        return Phh

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
        return table1

    # ----------------------------EXPERIMENT Clàssic-------------------------------
    def rho(self, photon, alpha, beta):
        # fem passar el fotó pels polaritzadors
        detector1 = 0
        detector2 = 0
        photon = photon % math.pi
        alpha = alpha % math.pi
        beta = beta % math.pi
        if abs(alpha - photon) <= (math.pi / 4) or abs(alpha - photon) > (
                3 * math.pi / 4):  # si passa pel primer pol
            detector1 = 1
        if abs(beta - photon) <= (math.pi / 4) or abs(beta - photon) > (3 * math.pi / 4):  # si passa pel segon
            detector2 = 1
        detectors = [detector1, detector2]
        return detectors

    def rho2(self, photon, alpha, beta):
        detector1 = 0
        detector2 = 0
        photon = photon % math.pi
        alpha = alpha % math.pi
        beta = beta % math.pi
        if random.uniform(0, 1) < math.cos(alpha - photon) ** 2:
            detector1 = 1
        if random.uniform(0, 1) < math.cos(beta - photon) ** 2:
            detector2 = 1
        detectors = [detector1, detector2]
        return detectors

    def hvt(self, alpha, beta, rho_select):
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
        for alph in pol1:
            for bet in pol2:
                # nombre total de fotons tirats, vertical-vertical, VH...
                VV = 0
                HH = 0
                HV = 0
                VH = 0
                for i in range(1, self.photons):
                    # fotó de polarització aleatòria entre [-pi,pi]
                    photon = random.uniform(-math.pi,
                                            math.pi)  # ha de ser entre -pi i pi perquè si no el alpha=-45 no pilla cap al detector1

                    # fem passar el fotó pels polaritzadors
                    if rho_select == 0 or rho_select == 2:
                        detectors = self.rho(photon, alph, bet)  # funció de distribució de probabilitat
                    elif rho_select == 1:
                        detectors = self.rho2(photon, alph, bet)

                    detector1 = detectors[0]
                    detector2 = detectors[1]
                    # organitzem els resultats per VV, VH, HV, HH
                    if detector1 == 1 and detector2 == 1:
                        VV = VV + 1
                    elif detector1 == 1 and detector2 == 0:
                        VH = VH + 1
                    elif detector1 == 0 and detector2 == 1:
                        HV = HV + 1
                    elif detector1 == 0 and detector2 == 0:
                        HH = HH + 1
                    else:
                        print('error', detector1, detector2)

                # [angle del pol1, angle pol2, deteccions detector 1, deteccions detector 2, Entrellaçats (detectats VV)]
                resultats = [alph, bet, VH + VV, HV + VV, VV]
                table1.append(resultats)
                resultats = []
        return table1

    def scalc(self, exp_select, alpha, beta, rho_select):
        if exp_select == 0:  # qua
            table1 = self.expqua(alpha, beta)
        elif exp_select == 1:  # HVT
            table1 = self.hvt(alpha, beta, rho_select)
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

        S = Elist[0] - Elist[1] + Elist[2] + Elist[3]
        return S

    scalcvec = np.vectorize(scalc)

    def sigma(self, alpha, beta):
        # Nc=llista de coincidències, agafa la última columna de la taula
        table1 = self.expqua(alpha, beta)
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
        return (sigma)

    # Calcs S from real data

    def s_calc_data(self, table1):
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
        S = Elist[0] - Elist[1] + Elist[2] + Elist[3]
        return S

    def sigma_data(self, table1):
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
        return (sigma)

    def sweepS(self):
        res = []
        # we take the lowest of b1 and b2
        alphalist = np.linspace(0, 2 * np.pi, 200)
        if self.b1 > self.b2:
            angle1 = self.b2
            angle2 = self.b1
        else:
            angle1 = self.b1
            angle2 = self.b2
        # WARNING: max beta linspace points ~ 50
        betalist = np.linspace(angle1, angle2, 100)

        coords = (alphalist, betalist)
        return coords
