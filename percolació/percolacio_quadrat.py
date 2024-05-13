import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
from scipy import stats
import math as math




##########################################################################################################################

#IN: 1) Mida n de la matriu. 2) Probabilitat p de tenir 1 a una posició determinada
#OUT: Matriu nxn de 1's i 0's

def matriu_quadrat(n,p):
    matriu = np.zeros((n,n),int)    #Comencem generant una matriu buida de dimensions nxn

    for i in range(n):
        for j in range(n):
            q = random.random() #Probabilitat de formar enllaç, si  q >= p es forma enllaç (1 a la matriu), altrament 0
            if q < p:
                matriu[i,j] = 1
    return matriu

############################################################################################################################

#prova

def matriu_aux(n):
    return np.random.randint(2, size=(n,n))



###########################################################################################################################


#Esta función emplea el algoritmo bfs para encontrar caminos a partir de un punto dado.
#Devuelve un vector con los puntos de la matriz visitados
#Inici és una tupla de dos números que indiquen una posició dins la matriu.
def bfs(matriu, inici, visitat):
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

def busca_clusters(matriu):

    n = len(matriu[0])
    visitat = np.full((n, n), False, dtype=bool)
    llista_clusters = []

    for i in range(n):
        for j in range(n):
            if matriu[i, j] == 1 and not visitat[i, j]:
                llista_clusters.append(bfs(matriu, (i, j), visitat))
                visitat[i, j] = True

    return llista_clusters

###########################################################################################################################

#IN: 1) Matriu quadrada nxn. 2) Llista dels clusters
#OUT: Matriu amb els diferents clusters identificats amb diferents números

def pintar_clusters(matriu,vertex_actius):

    matriu_pintada = matriu.copy()
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

def percola(matriu, cluster_per):

    clusters = busca_clusters(matriu)
    print(clusters)
    # # de clusters del sistema
    n = len(matriu[0]) #variable mal inicialitzada, per això no acabava de funcionar el programa
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

#IN: 1) Matriu de dimensions nxn
#OUT: Fracció de vèrtexs connectats

def fraccio_ver_con(matriu):

    ver_total = len(matriu)*len(matriu[0])
    clusters = busca_clusters(matriu)
    count_ver = set()
    count = 0

    for cluster in clusters:
        if len(cluster) > 1:
            for vertex in cluster:
                if vertex not in count_ver:
                    count_ver.add(vertex)
                    count += 1

    return count/ver_total

###########################################################################################################################

#IN: 1) Matriu de dimensions nxn
#OUT: Fracció de la mida del cluster més gran

def biggest_cluster_frac(matriu):

    ver_total = len(matriu)*len(matriu[0])
    clusters = busca_clusters(matriu)
    max_cluster_size = 0

    for cluster in clusters:
        cluster_size = len(cluster)
        #actualitzem la mida del cluster més gran fins el moment
        if cluster_size > max_cluster_size:
             max_cluster_size = cluster_size

    return max_cluster_size/ver_total

###########################################################################################################################

#IN: 1) Matriu de dimensions nxn
#OUT: Fracció del cluster que ha percolat (en cas d'haver percolació). Altrament retorna 0.

def fracc_cluster_per(matriu):

    n_total = len(matriu)*len(matriu[0])
    aux = []
    if percola(matriu, aux) == -1:
        return 0
    else:
        clusters = busca_clusters(matriu)[percola(matriu, aux) - 1]
        n_clusters_per = len(clusters)
        return n_clusters_per/n_total

############################################################################################################################

#IN: 1) Iteracions de la simulació. 2) Mida n de les matrius
#OUT: 2) Vector de dimensions 'interacions' amb un conjunt de punts crítics
#per matrius de mida nxn
def vector_pc(iteracions,n):
    x = np.zeros(iteracions)
    for i in range(iteracions):
        p = 0
        matriu = matriu_quadrat(n, p)
        while percola(matriu,[]) == -1:
            matriu = matriu_quadrat(n, p)
            p += 0.01
            x[i] = p
    return x

############################################################################################################################
############################################################################################################################

                                            # ESTADÍSTICA:

#IN: 1) Vector de valors. 2) n, Número d'iteracions
#OUT: Desviació estàndard
def standard_dev(n,valors):
    suma = 0
    mean = mean_value(valors)
    for i in range(n):
        suma += (valors[i] - mean)**2
    return np.sqrt(suma/(n-1))

############################################################################################################################

#IN: 1) Vector values
#OUT: Mitjana de values (sumar els seus elements i dividr-los per la mida de values
def mean_value(values):
    suma = sum(values)
    return suma/len(values)

############################################################################################################################

#IN: 1) Vector de valors. 2) n, Número d'iteracions
#OUT: Incertesa de la mesura
def incertesa(n,valors):
    desv_est = standard_dev(n,valors)
    return desv_est/np.sqrt(n)

############################################################################################################################

#IN: 1) Vector de valors. 2) n, Número d'iteracions
#OUT: Incertesa relativa
def incertesa_relativa(n,valors):
    return (incertesa(n,valors)/mean_value(valors))

############################################################################################################################

def U_n(valors,iteracions):
#Aplicar la fórmula 7 del paper:
#Investigations into the Influence of Matrix Dimensions and Number of Iterations on the Percolation Phenomenon for Direct Current
    mean = mean_value(valors)
    standard_dev(iteracions,valors)
    suma = 0
    for i in range(iteracions):
        suma += (x[i]+mean)
    numerador = (1/iteracions)*suma
    denominador = standard_dev(iteracions,valors)/np.sqrt(iteracions)
    return numerador/denominador

############################################################################################################################
############################################################################################################################

                                                # GRÀFIQUES:

#Gràfica del punt crític de percolació en funció de les dimensions de la matriu quadrada pel cas site percolation
def g1():
    # Gràfica que mostra el punt crític de percolació en funció de les dimensions de la matriu
    # Per cada probabilitat dins de probabilities generem matrius des de la mida inferior fins
    # la superior (elements dins de matrix_size) i busquem el punt de percolació amb percola.
    # Emmagatzemem el punt de percolació i mostrem els resultats en una gràfica

    probabilities = np.arange(0.1, 1.0, 0.01)
    matrix_sizes = range(2, 61)  # Mides de matrius que anem a provar
    num_trials = 1000
    aux = []

    critical_probabilities = {size: None for size in matrix_sizes}

    for size in matrix_sizes:
        for p in probabilities:
            percolating_trials = 0
            matrix = matriu_quadrat(size, p)
            if percola(matrix,aux) != -1:
                percolating_trials += 1
            if percolating_trials / 1 > 0:  # Percola, afegim el punt
                critical_probabilities[size] = p
                break
    # Regressió polinòmica per veure possibles tendències
    x_values = np.array(list(critical_probabilities.keys()))
    y_values = np.array(list(critical_probabilities.values()))
    regression_coeffs = np.polyfit(x_values, y_values, 2)  # Adjust degree as needed
    regression_line = np.poly1d(regression_coeffs)
    plt.plot(x_values, y_values, marker='o', label='Data')
    plt.plot(x_values, regression_line(x_values), linestyle='--', color='red', label='Regression')

    plt.plot(matrix_sizes, list(critical_probabilities.values()), marker='o')
    plt.xlabel('Mida de la matriu')
    plt.ylabel('Punt crític de percolació')
    plt.title('Punt crític de percolació en funció de la mida de la matriu')
    plt.grid(True)
    plt.show()

###########################################################################################################################

#Gràfica de la fracció de vèrtexs connectats en funció de la probabilitat de percolació pel cas site percolation
def g2():
    probabilities = np.arange(0.1, 1.0, 0.1)
    num_trials = 100  # Número de matrices a generar para cada probabilidad
    avg_fracs = []  # Almacenar el promedio de las fracciones de vértices conectados

    for p in probabilities:
        frac_sum = 0
        for _ in range(num_trials):
            matrix = matriu_quadrat(50, p)
            frac_sum += biggest_cluster_frac(matrix)  #fracció de cluster més gran ESTO ESTÁ MAL, DEBERÍA SER LA FRACCIÓN DE CLUSTERS CONECTADOS INDEPENDIENTEMENTE DE SU TAMAÑO
        avg_frac = frac_sum / num_trials  #mitjana de cluster de mida més gran per matriu de mida n i p
        avg_fracs.append(avg_frac)

    #Graficar els resultats
    plt.plot(probabilities, avg_fracs)
    plt.xlabel('p')
    plt.ylabel('Fracció del cluster més gran')
    plt.title('Fracció del cluster més gran en funció de p')
    plt.grid(True)
    plt.show()

############################################################################################################################

# Per dibuixar el mapa
def heatmap(n,p,iteracions):
    matriu_diagrama = np.zeros((n, n), dtype=int)
    for k in range(iteracions):
        matriu = matriu_quadrat(n, p)
        for i in range(len(matriu)):
            for j in range(len(matriu[0])):
                if (matriu[i][j]) == 1:
                    matriu_diagrama[(n - 1) - i][j] += 1

    plt.imshow(matriu_diagrama, cmap='viridis', interpolation=None)
    plt.colorbar()
    plt.title('Mapa de distribució aleatòria dins de les matrius')
    plt.xlabel('Columnes (0-55)')
    plt.ylabel('Files (0-55)')
    plt.gca().invert_yaxis()
    plt.show()

############################################################################################################################

def figure5(iteracions):
    ps = [0.2, 0.6, 0.9]
    ns = [55, 101, 155]
    for p in ps:
        for n in ns:
            vector = np.zeros(n)
            centre = (n - 1) // 2
            for k in range(iteracions):
                matriu = matriu_quadrat(n, p)
                for i in range(n):
                    if (matriu[i][centre] == 1 or matriu[centre][i] == 1):
                        vector[i] += 1

            plt.scatter(range(n), vector, label=f'n={n}, p={p}')  # Etiquetar cada conjunto de datos con n y p

    plt.xlabel('Coordinates of nodes, a.u.')
    plt.ylabel('Number of points, a.u.')
    plt.title('Evolution of vectors for different n and p')
    plt.grid(True)
    plt.legend()
    plt.show()

############################################################################################################################

def histograma(n,iteracions):
    #primer de tot anem a obtenir un vector de dimensió n on anirem
    #guardant els valors de percolació de cada iteració n
    x = vector_pc(iteracions,n)
    n_bins = 7
    hist = np.histogram(x,bins=n_bins,density=True)
    #hist_dist = stats.rv_histogram(histogram=hist,density=True)
    hist_dist = stats.rv_histogram(histogram=hist, density=True)
    plt.hist(x,bins=n_bins,edgecolor="white",density=True,label="Histogram")
    eix_x = np.linspace(min(x),max(x),n_bins)
    plt.plot(eix_x,hist_dist.pdf(eix_x),label="PDF")

    plt.xlabel('Punts crítics')
    plt.ylabel('Densitat')
    plt.title('Histograma de Freqüències')

    plt.grid(True)
    plt.show()

############################################################################################################################
############################################################################################################################

                                # CÓDIGO PRINCIPAL:

#condicions incials
n = 151
p = 0.7
iteracions = 5 * 10 ** 4

# (1): Gràfica del punt crític de percolació en funció de les dimensions de la matriu quadrada pel cas site percolation
#g1()

# (2): Gràfica de la fracció de vèrtexs connectats en funció de la probabilitat de percolació
#g2()

# (3): mapa que mostra la distribució aleatòria de generació de números dins de la matrius (per diferents mides i p's)
#al simular diferents mides i p's no s'ha observat cap patró en cap cas. Per tant, les distribucions aleatòries
# es generen correctament. (Figure 2, 3 i 4)
#heatmap()

# (4): 1's al llarg de les línies horitzontals i verticals que passen pel centre
#figure5(iteracions)



# (6) Histograma
histograma(n,iteracions)

#calcul punt crític
x = vector_pc(iteracions,n)
mean = mean_value(x)
inc = incertesa(iteracions,x)
print('p_c = ', mean, ' +- ', inc)



############################################################################################################################

                                # PRUEBAS ETC:

n = 5
p = 0.2

matriu = matriu_quadrat(n,p)
cluster_vec = []
visitat = np.full((n, n), False, dtype=bool)
print(matriu)
trobat = False
for i in range(n):
    for j in range(n):
        if matriu[i,j] == 1 and not trobat:
            #cluster_vec = cluster_funcio(matriu, i, j, n,visitat)
            trobat = True
    if trobat:
        break

print(cluster_vec)
print(busca_clusters(matriu))
print(len(busca_clusters(matriu)))
caca = busca_clusters(matriu)
print(pintar_clusters(matriu,caca))
cluster_percolat = []
num = percola(matriu,cluster_percolat)
if num != -1:
    print(num)
    print(cluster_percolat)
else:
    print('No ha percolat :(')

print(fraccio_ver_con(matriu))
print(biggest_cluster_frac(matriu))
print('fracció del cluster que ha percolat')
print(fracc_cluster_per(matriu))


