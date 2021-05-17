def qubitnumber():
    return 2

def gate():
    string="""
    circuit.h(q0)
    circuit.h(q1)
    circuit.t(q0)
    circuit.s(q1)
    circuit.h(q0)"""
    return string