# -*- coding: utf-8 -*-

import numpy as np

global matrix
matrix =np.empty((3,4), dtype="<U5")
matrix[1][2]='CCX'
matrix[0][2]='CCX'
matrix[2][2]='CCX'

matrix[0][0]='H'
matrix[2][0]='H'

matrix[0][1]='CX'
matrix[1][1]='CX'

gates_2={'CX': 2, 'CCX': 3}


multigates=[]
test=[1,0,2,2]
multigates.append(test)

test=[1,0,1]
multigates.append(test)

def QiskitConverter(matrix, multigates, row_name, gates_2):
    
    File = open("QiskitCircuit.txt","w")

    File.write('from qiskit import * \n')
    File.write('import matplotlib.pyplot as plt \n \n')
    File.write('qr = QuantumRegister({}) \n'.format(matrix.shape[0]))
    File.write('cr = ClassicalRegister({}) \n'.format(matrix.shape[0]))
    File.write('circuit = QuantumCircuit(qr, cr) \n \n')
    #File.write('%matplotlib inline \n \n')
    
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            if matrix[i][j] != '':
                if matrix[i][j] in gates_2:
                    for k in multigates:
                        if k[-1]==j and k[0]==i:
                            text='circuit.{}('.format(matrix[i][j].lower())
                            for t in range(len(k)-2):
                                text='{}{}'.format(text,'qr[{}], '.format(k[t+1]))
                                
                            text='{}{} \n'.format(text,'qr[{}])'.format(i))
                            File.write(text)
                            print(text)
                            
                else:
                    File.write('circuit.{}(qr[{}]) \n'.format(matrix[i][j].lower(),i))
                    
                    
                    
    File.write('\ncircuit.measure(qr, cr) \n')
    
    File.write("simulator = Aer.get_backend('qasm_simulator') \n")
    File.write('result = execute(circuit, backend = simulator).result() \n \n')
    
    File.write('from qiskit.tools.visualization import plot_histogram \n')
    File.write("plot_histogram(result.get_counts(circuit)) \n")
    File.write('plt.show() \n')
                


    



    File.close()
 
    
print(matrix)
QiskitConverter(matrix, multigates,0,gates_2)