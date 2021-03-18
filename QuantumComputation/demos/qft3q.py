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

    crot = QuantumCircuit(QuantumRegister(1)) 
    crot.u3(1.5707963267948966, 0.0, 0, [0])
    circuit.mcmt(crot, [1], [2])

    crot = QuantumCircuit(QuantumRegister(1)) 
    crot.u3(0.7853981633974483, 0, 0, [0])
    circuit.mcmt(crot, [0], [2])
    circuit.h(qr[1]) 

    crot = QuantumCircuit(QuantumRegister(1)) 
    crot.u3(1.5707963267948966, 0.0, 0, [0])
    circuit.mcmt(crot, [0], [1])
    circuit.h(qr[0]) 
    circuit.swap(qr[2], qr[0]) 

    plotresults = execute(circuit,backend).result().get_statevector()
    circuit.u3(-0, -0, 0, qr[0])
    circuit.u3(-0, -0, 0, qr[1])
    circuit.u3(-0, -0, 0, qr[2])
    results = execute(circuit,backend).result().get_statevector()

    return results, plotresults

def gatesmatrix():
    matrix=[
    ['', '', 'CROT', '', '', 'CROT', '', 'H', 'SWAP', '', '', ''],
    ['', 'CROT', '', '', 'H', 'CROT', '', '', '', '', '', ''],
    ['H', 'CROT', 'CROT', '', '', '', '', '', 'SWAP', '', '', '']]

    shape=(3, 12)

    multigates=[[[1], [2], [1.5707963267948966, 0.0, 'CROT'], 1], [[0], [1], [1.5707963267948966, 0.0, 'CROT'], 5], [0, 2, 8], [[0], [2], [0.7853981633974483, 0, 'CROT'], 2]]

    return shape, matrix, multigates

def name():
    return "QUANTUM FOURIER TRANSFORM (3 QUBITS)"