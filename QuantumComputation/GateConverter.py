# -*- coding: utf-8 -*-
import numpy as np
import os

def ImportGate(qubitnumber, string, cgrid, matrix, gates_2):
    cgrid.clear()
    matrix =np.empty((0,0), dtype="<U7")
    multigates=[]
    
    for i in range(qubitnumber):
        cgrid.add_row()
        matrix=np.append(matrix, np.empty((1,matrix.shape[1]), str), axis=0)
    lines=string.splitlines()
    for line in lines:
        if 'circuit' in line:
            cgrid.add_col()
            matrix=np.append(matrix, np.empty((matrix.shape[0],1), str), axis=1)
            col=matrix.shape[1]-1
            command=line.split('.')[1]
            gate=command.split('(')[0]
            qubits=command.split('(q')[1][0:-1]
            
            if gate.upper() in gates_2:
                qbitlist=qubits.split(", q")
                qbits=[]
                for i in range(len(qbitlist)):
                    qbits.append(int(qbitlist[i]))
                    matrix[qbits[i]][col]=gate.upper()
                qbits.reverse()
                qbits.append(col)
                multigates.append(qbits)
                
                
            else:
                qbits=int(qubits)
                
                matrix[qbits][col]=gate.upper()
    
    if matrix.shape[1] < 12:
        for i in range(12-matrix.shape[1]):
            cgrid.add_col()
            matrix=np.append(matrix, np.empty((matrix.shape[0],1), str), axis=1)
    
    return matrix, multigates

def UpdateGate(gatename, qubitnumber, matrix, multigates, gates_2):
    File = open(os.path.dirname(os.path.realpath(__file__))+"\cgates\\"+gatename+'.py',"w")
    s='    '

    File.write('def qubitnumber():\n')
    File.write(s+'return '+str(qubitnumber))

    File.write('\n\ndef gate():\n')
    File.write(s+'string="""')
    
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            if matrix[i][j] != '':
                if matrix[i][j] in gates_2:
                    for k in multigates:
                        if k[-1]==j and k[0]==i:
                            text='circuit.{}('.format(matrix[i][j].lower())
                            for t in range(len(k)-2):
                                text='{}{}'.format(text,'q{}, '.format(k[t+1]))
                                
                            text='{}{}'.format(text,'q{})'.format(i))
                            File.write('\n'+s+text)
                else:
                    File.write('\n'+s+'circuit.{}(q{})'.format(matrix[i][j].lower(),i))
    
    File.write('"""\n')
    File.write(s+'return string')

    File.close()
    return
                
            
        
        