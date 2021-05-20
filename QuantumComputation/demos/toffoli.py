from qiskit import * 
from qiskit.quantum_info import Statevector 
import numpy as np 
from numpy import pi 
import matplotlib.pyplot as plt 

def GetStatevector():
    backend = Aer.get_backend('statevector_simulator')
    qr = QuantumRegister(3) 
    cr = ClassicalRegister(3) 
    circuit = QuantumCircuit(qr) 
 
    circuit.h(qr[2]) 
    circuit.cx(qr[1], qr[2]) 
    circuit.tdg(qr[2]) 
    circuit.cx(qr[0], qr[2]) 
    circuit.t(qr[2]) 
    circuit.cx(qr[1], qr[2]) 
    circuit.tdg(qr[2]) 
    circuit.cx(qr[0], qr[2]) 
    circuit.t(qr[1]) 
    circuit.t(qr[2]) 
    circuit.h(qr[2]) 
    circuit.swap(qr[2], qr[1]) 
    circuit.cx(qr[0], qr[2]) 
    circuit.t(qr[0]) 
    circuit.tdg(qr[2]) 
    circuit.cx(qr[0], qr[2]) 
    circuit.swap(qr[2], qr[1]) 

    plotresults = execute(circuit,backend).result().get_statevector()
    circuit.u3(-0, -0, 0, qr[0])
    circuit.u3(-0, -0, 0, qr[1])
    circuit.u3(-0, -0, 0, qr[2])
    results = execute(circuit,backend).result().get_statevector()

    return results, plotresults

def gatesmatrix():
    matrix=[
    ['', '', '', '', '', '', 'CX', '', '', '', 'CX', '', '', '', 'CX', 'T', 'CX', '', '', ''],
    ['', '', '', '', 'CX', '', '', '', 'CX', '', '', 'T', '', 'SWAP', '', '', '', 'SWAP', '', ''],
    ['', '', '', 'H', 'CX', 'Tdg', 'CX', 'T', 'CX', 'Tdg', 'CX', 'T', 'H', 'SWAP', 'CX', 'Tdg', 'CX', 'SWAP', '', '']]

    shape=(3, 20)

    multigates=[[2, 1, 4], [2, 0, 6], [2, 1, 8], [2, 0, 10], [1, 2, 13], [2, 0, 14], [2, 0, 16], [1, 2, 17]]

    return shape, matrix, multigates

def name():
    return 'TOFFOLI GATE'

def img_num():
    return 1


def highlight(page):
    grid=np.zeros((3,20))
    return(grid)
    
def customize(screen):
    screen.title.text="Toffoli Gate"