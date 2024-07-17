import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
from scipy import stats

class ClassPercolacioQuadrat:

    #Constructor de la classe. S'executa automàicament al crear una nova instància
    #de la classe.
    def __init__(self,n,p):
        self.n = n
        self.p = p
        self.matriu = self.matriu_quadrat(n,p)
        self.visitat = np.full((n,n),False,dtype=bool)
        self.cluster = []
        self.values = []

##########################################################################################################################

    #IN: 1) Mida n de la matriu. 2) Probabilitat p de tenir 1 a una posició determinada
    #OUT: Matriu nxn de 1's i 0's

    def matriu_quadrat(self,n,p):
        matriu = np.zeros((n,n),int)    #Comencem generant una matriu buida de dimensions nxn

        for i in range(n):
            for j in range(n):
                q = random.random() #Probabilitat de formar enllaç, si  q >= p es forma enllaç (1 a la matriu), altrament 0
                if q < p:
                    matriu[i,j] = 1
        return matriu


    def matriu_factal(self,iteracions,p):

        n = 3**iteracions

        for i in range(n):
            for j in range(n):
                if self.matriu[i,j] == 1:
                    q = random.random()
                    if q < p:
                        self.matriu[i,j] = 1
                    else:
                        self.matriu[i,j] = 0
        return self



###########################################################################################################################


    #Esta función emplea el algoritmo bfs para encontrar caminos a partir de un punto dado.
    #Devuelve un vector con los puntos de la matriz visitados
    #Inici és una tupla de dos números que indiquen una posició dins la matriu.

    def bfs(self,matriu, inici, visitat):
        #visitat = np.zeros(matriu.shape, dtype=bool)
        queue = deque([inici])
        vertex_actius = []

        #Mientras queden elementos en la cola para ser explorados el bucle no para
        #El concepto de deque es una abstracción de estructura de datos que sigue el principio fifo.
        while queue:
            row, col = queue.popleft()
            visitat[row, col] = True
            vertex_actius.append((row, col))

            # Obtener vecinos válidos no visitados
            for neighbor_row, neighbor_col in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
                #amb la condició 0 <= ens assegurem que estem dins dels límits de la matriu.
                if 0 <= neighbor_row < len(matriu) and 0 <= neighbor_col < len(matriu[0]) and not visitat[neighbor_row, neighbor_col] and matriu[neighbor_row, neighbor_col] == 1:
                    visitat[neighbor_row, neighbor_col] = True
                    #vertex_actius.append((neighbor_row,neighbor_col))
                    queue.append((neighbor_row, neighbor_col))

        return vertex_actius

############################################################################################################################


    #IN: 1) Matriu quadrada de dimensions nxn
    #OUT: Llista amb tots els clusters de la matriu. Utilitza la funció cluster_funcio per buscar el
    # cluster que comença a la posició i j

    #Ara funciona amb bfs en comptes d'utilitzar la funció cluster_funcio (petava per matrius de mida >= 60

    def busca_clusters(self):

        n = len(self.matriu[0])
        visitat = np.full((n, n), False, dtype=bool)
        llista_clusters = []

        for i in range(n):
            for j in range(n):
                if self.matriu[i, j] == 1 and not visitat[i, j]:
                    llista_clusters.append(self.bfs(self.matriu, (i, j), visitat))
                    visitat[i, j] = True

        return llista_clusters

###########################################################################################################################

    #IN: 1) Matriu quadrada nxn. 2) Llista dels clusters
    #OUT: Matriu amb els diferents clusters identificats amb diferents números

    def pintar_clusters(self,vertex_actius):

        matriu_pintada = self.matriu.copy()
        for k in range(len(vertex_actius)):
            for i in range(len(vertex_actius[k])):
                x = vertex_actius[k][i][0]
                y = vertex_actius[k][i][1]
                matriu_pintada[x,y] = k + 1
        return matriu_pintada

###########################################################################################################################

    #IN: 1) Matriu de dimensions nxn
    #    2) cluster_per: llista inicialment buida. Si hi ha percolació es guardaran aquí
    #       les posicions del cluster que percola
    #OUT: Un número natural que indica quin és el cluster que ha percolat. Si no hi ha percolació es retorna -1

    def percola(self):

        clusters = self.busca_clusters()
        print(clusters)
        # # de clusters del sistema
        n = len(self.matriu[0]) #variable mal inicialitzada, per això no acabava de funcionar el programa
        #ens servirà per identificar el cluster que percola. Els enters no poden canviar si són passats com arguments de la funció
        perc_n = 0

        #Anem cluster per cluster veient si hi ha algun punt a la part superior, inferior
        #esquerra o dreta. Posem les variables corresponents a True i a partir de veure si
        #hi han vores oposades amb valors True determinem la percolació (o no percolació)

        top = False
        bottom = False
        left = False
        right = False

        for k in range(len(clusters)):
            top = False
            bottom = False
            for i in range(len(clusters[k])):
                if clusters[k][i][0] == 0:
                    top = True
                    perc_n = k
                if clusters[k][i][0] == n - 1:
                    bottom = True
                    perc_n = k
                if clusters[k][i][0] == n - 1:
                    left = True
                    perc_n = k
                if clusters[k][i][1] == n - 1:
                    right = True
                    perc_n = k
            if (top and bottom):
                break

            #comprovem totes les possibilitats de percolació

        #if ((top and bottom) or (top and right) or (top and left) or (left and right) or (bottom and right) or (bottom and left)):
            #clusters_per = cluster_percolat.append(clusters[perc_n])
        if (top and bottom):
            print('ha percolat!')
            perc_n += 1
        else:
            perc_n = -1

        return perc_n

###########################################################################################################################

    #IN: Índex d'un cluster
    #OUT: Mida del cluster
    def mida_cluster(self,cluster_index):
        mida = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.matriu[i,j] == cluster_index:
                    mida += 1
        return mida

###########################################################################################################################

    def biggest_cluster_frac(self):

        ver_total = len(self.matriu) * len(self.matriu[0])
        clusters = self.busca_clusters()
        max_cluster_size = 0

        for cluster in clusters:
            cluster_size = len(cluster)
            # actualitzem la mida del cluster més gran fins el moment
            if cluster_size > max_cluster_size:
                max_cluster_size = cluster_size

        return max_cluster_size/ver_total

    ###########################################################################################################################
                                                        #ESTADÍSTICA

    # IN: 1) Vector values
    # OUT: Mitjana de values (sumar els seus elements i dividr-los per la mida de values
    @staticmethod
    def mean_value(values):
        suma = sum(values)
        return suma / len(values)


    # IN: 1) Vector de valors. 2) n, Número d'iteracions
    # OUT: Incertesa de la mesura
    @staticmethod
    def incertesa(values):
        desv_est = ClassPercolacioQuadrat.standard_dev(values)
        return desv_est/np.sqrt(len(values))


    #IN: 1) Vector de valors. 2) n, Número d'iteracions
    #OUT: Desviació estàndard
    @staticmethod
    def standard_dev(values):
        suma = 0
        mean = ClassPercolacioQuadrat.mean_value(values)
        for i in range(len(values)):
            suma += (values[i] - mean)**2
        return np.sqrt(suma/(len(values)-1))
























