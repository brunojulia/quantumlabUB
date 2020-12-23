def qubitnumber():
    return 3

def gate():
    string="""
    circuit.h(q2)
    circuit.cu1(pi/2, q1, q2)
    circuit.cu1(pi/4, q0, q2)
    circuit.h(q1)
    circuit.cu1(pi/2, q0, q1)
    circuit.h(q0)
    circuit.swap(q0,q2) """
    
    return string


    