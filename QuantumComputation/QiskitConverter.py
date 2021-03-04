# -*- coding: utf-8 -*-
import CustomGates as cgates


def QiskitConverter(matrix, multigates, row_name, gates_2, customgates, angles):
    File = open("QiskitCircuit.py","w")
    s='    '

    File.write('from qiskit import * \n')
    File.write('from qiskit.quantum_info import Statevector \n')
    File.write('import numpy as np \n')
    File.write('from numpy import pi \n')
    File.write('import matplotlib.pyplot as plt \n\n')
    #File.write('from qiskit.visualization import plot_histogram\n\n')

    File.write('def GetStatevector():\n')
    
    #File.write(s+'global state \n')
    File.write(s+"backend = Aer.get_backend('statevector_simulator')\n")
    
    File.write(s+'qr = QuantumRegister({}) \n'.format(matrix.shape[0]))
    File.write(s+'cr = ClassicalRegister({}) \n'.format(matrix.shape[0]))
    File.write(s+'circuit = QuantumCircuit(qr) \n \n')
    

    
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
                            File.write(s+text)
                elif matrix[i][j] in customgates:
                    for k in multigates:
                        if k[-1]==j and k[0]==i:
                            text=getattr(cgates, matrix[i][j]).gate()
                            for qubit in range(len(k)-1):
                                before='q'+str(qubit)
                                after='qr['+str(k[qubit])+']'
                                text=text.replace(before, after)+'\n'
                            
                            
                            File.write(s+text)
                else:
                    File.write(s+'circuit.{}(qr[{}]) \n'.format(matrix[i][j].lower(),i))
    
    File.write('\n'+s+'plotresults = execute(circuit,backend).result().get_statevector()\n')
    for i in range(matrix.shape[0]):
        string=angles[i][0]+', '+angles[i][1]+', 0, qr[{}])\n'.format(i)
        File.write(s+'circuit.u3('+string)
                    
                    
    File.write(s+'results = execute(circuit,backend).result().get_statevector()\n\n')
    #File.write(s+'state = Statevector.from_instruction(circuit)\n')
    File.write(s+'return results, plotresults\n\n')


    File.write('def gatesmatrix():\n')
    File.write(s+'matrix=[\n')
    rows=[]
    str1= ',\n'+s
    str2="', '"
    for i in matrix:
        string="['"+str2.join(i)+"']"
        rows.append(string)
        
    stringmatrix=s+str1.join(rows)+']\n\n'
    File.write(stringmatrix)
    File.write(s+'shape='+str(matrix.shape)+'\n\n')
    
    multi=[]
    str3=', '
    for i in multigates:
        string="["+str3.join([str(elem) for elem in i])+"]"
        multi.append(string)
    File.write(s+'multigates=['+str3.join(multi)+']\n\n')
    
    File.write(s+'return shape, matrix, multigates\n')

    File.close()
    return
