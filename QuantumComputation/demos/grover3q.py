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
 
    circuit.h(qr[0]) 
    circuit.h(qr[1]) 
    circuit.h(qr[2]) 

    crot = QuantumCircuit(QuantumRegister(1)) 
    crot.u3(0.0, 3.141592653589793, 0, [0])
    circuit.mcmt(crot, [2], [0])

    crot = QuantumCircuit(QuantumRegister(1)) 
    crot.u3(0.0, 3.141592653589793, 0, [0])
    circuit.mcmt(crot, [1], [0])
    circuit.h(qr[0]) 
    circuit.h(qr[1]) 
    circuit.h(qr[2]) 
    circuit.x(qr[0]) 
    circuit.x(qr[1]) 
    circuit.x(qr[2]) 

    crot = QuantumCircuit(QuantumRegister(1)) 
    crot.u3(0.0, 3.141592653589793, 0, [0])
    circuit.mcmt(crot, [2, 1], [0])
    circuit.x(qr[0]) 
    circuit.x(qr[1]) 
    circuit.x(qr[2]) 
    circuit.h(qr[0]) 
    circuit.h(qr[1]) 
    circuit.h(qr[2]) 

    plotresults = execute(circuit,backend).result().get_statevector()
    circuit.u3(-0, -0, 0, qr[0])
    circuit.u3(-0, -0, 0, qr[1])
    circuit.u3(-0, -0, 0, qr[2])
    results = execute(circuit,backend).result().get_statevector()

    return results, plotresults

def gatesmatrix():
    matrix=[
    ['H', '', 'CROT', 'CROT', '', 'H', 'X', 'CROT', 'X', 'H', '', ''],
    ['H', '', '', 'CROT', '', 'H', 'X', 'CROT', 'X', 'H', '', ''],
    ['H', '', 'CROT', '', '', 'H', 'X', 'CROT', 'X', 'H', '', '']]

    shape=(3, 12)

    multigates=[[[2], [0], [0.0, 3.141592653589793, 'Z'], 2], [[1], [0], [0.0, 3.141592653589793, 'Z'], 3], [[2, 1], [0], [0.0, 3.141592653589793, 'Z'], 7]]

    return shape, matrix, multigates

def name():
    return "GROVER'S ALGORITHM (3 QUBITS)"



def img_num():
    return 1


def highlight(page):
    grid=np.zeros((3,12))
    return(grid)
    
def customize(screen):
    screen.title.text="Grover's Agorithm (3 Qubits)"