from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
import numpy as np
import matplotlib.pyplot as plt
from class_percolacio_quadrat import ClassPercolacioQuadrat


class SierpinskiCarpetWidget(Widget):
    def __init__(self, **kwargs):
        super(SierpinskiCarpetWidget, self).__init__(**kwargs)
        self.iteracions  # Nivell de recursió del fractal
        self.n = 3 ** self.iteracions  # mida d'un costat de la matriu. La matriu tindrà en total nxn cel·les
        self.p = 0.5  # probabilitat
        # inicialitzem una matriu buida de mida nxn amb 0's
        self.matrix = np.empty((self.n, self.n), int)
        # la plenem de 1's. D'aquesta manera soluciono el problema que tenia amb l'aparició
        # de números tan grans
        self.matrix.fill(1)
        # cridem aquesta funció per completar la seva matriu del fractal
        # variables per ordre d'aparició:
        # nivell de recursió del fractal, x, y, pos_x, pos_y, mida matriu
        # x y és des d'on comença a crear-se la matriu amb (0,0) a la cantonada inferior esquerra
        self.draw_carpet(self.iteracions, 0, 0, 0, 0, self.n)

        self.fractal_perc()
        # representació visual a partir de la matriu

        self.cluster_perc = []

        # comprovem si aquest fractal percola (pista, sí)
        perco = ClassPercolacioQuadrat(self.n, 0)
        perco.matriu = self.matrix
        resultat = perco.percola()
        if resultat != -1:
            print(print(f"El fractal percola en el cluster {resultat}"))
        else:
            print("El fractal no percola :(")

        self.paint_canvas()

    # funció recursiva per dibuixar el fractal (potser no és la manera més eficient)
    # a partir de de 10 iteracions quasi es mor l'ordinador
    def draw_carpet(self, iteracions, x, y, pos_x, pos_y, size):
        # cas base
        if iteracions == 0:
            self.matrix[pos_x, pos_y] = 1
        else:
            new_size = size//3  # ens quedem amb la part sencera de la divisió
            for i in range(3):
                for j in range(3):
                    # estem al centre
                    if i == 1 and j == 1:
                        # seleccionem una fracció del quadrat i la pintem de negre (0)
                        self.matrix[pos_x + i * new_size: pos_x + (i + 1) * new_size,
                                    pos_y + j * new_size: pos_y + (j + 1) * new_size] = 0

                    else:
                        self.draw_carpet(iteracions - 1, x + i * new_size, y + j * new_size,
                                         pos_x + i * new_size, pos_y + j * new_size, new_size)

    # a partir de la matriu de 0's i 1's pintem el canvas
    def paint_canvas(self):
        #comencem netejant el canvas
        self.canvas.clear()
        square_size = Window.height/self.n
        with self.canvas:
            for i in range(self.n):
                for j in range(self.n):
                    # pintem canvas
                        if self.matrix[i, j] == 1:
                            # blanc
                            Color(1, 1, 1, 1)
                        else:
                            # negre
                            Color(0, 0, 0, 1)

                        # mida i posició quadrat en qüestió
                        Rectangle(pos=(j * square_size, (self.n - i - 1) * square_size), size=(square_size, square_size))
        self.canvas.ask_update()

    def fractal_perc(self):
        # per generar la matriu de percolació a partir del fractal
        perco = ClassPercolacioQuadrat(self.n, self.p)
        perco.matriu = self.matrix
        perco.matriu_factal(self.iteracions, self.p)
        self.matrix = perco.matriu

    def find_percolation_threshold(self, num_iterations, p_start=0.0, p_end=1.0, p_step=0.01):
        percolation_probabilities = []

        for i in range (1,num_iterations+1):
            p = p_start
            self.n = mida = 3 ** i
            while p <= p_end:
                # Creem un nou fractal i actualitzem la matriu
                self.p = p
                self.matrix.fill(1)
                self.draw_carpet(num_iterations, 0, 0, 0, 0, self.n)
                self.fractal_perc()

                #Comprovem si percola
                perco = ClassPercolacioQuadrat(mida, p)
                perco.matriu = self.matrix
                if perco.percola() != -1:
                    percolation_probabilities.append(p)
                    break
                p += p_step



        #print(mean)
        return (percolation_probabilities)


    #Gràfica del punt de percolació en funció del número d'iteracions per generar el fractal
    #El número d'iteracions es deixa en 5 perquè a partir d'aquí comença a trigar moltíssim
    def plot_percolation_probabilities(self,num_iterations, p_start, p_end, p_step):
        probabilities = []
        perco_probs = self.find_percolation_threshold(num_iterations, p_start, p_end, p_step)

        plt.figure(figsize=(8, 6))
        plt.plot(num_iterations, perco_probs, marker='o', linestyle= 'None', color='b', label='Percolation Probability')
        plt.title('Punt crític p_c en funció de les iteracions per a Sierpinski Carpet')
        plt.xlabel('Iteracions')
        plt.ylabel('Probabilitat de percolació')
        plt.xticks(num_iterations)
        plt.grid(True)
        plt.legend()
        plt.show()



#classe per gestionar la simulació
class SierpinskiCarpetApp(App):
    def build(self):
        widget = SierpinskiCarpetWidget()
        print(widget.matrix)
        #widget.plot_percolation_probabilities(num_iterations=5, p_start=0.0, p_end=1.0, p_step=0.01)
        return widget



if __name__ == '__main__':
    SierpinskiCarpetApp().run()

























