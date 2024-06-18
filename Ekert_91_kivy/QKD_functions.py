import numpy as np

'''  QKD functions and variables '''

# Convention that Bob and Alice agree on in the preparation phase
MINUS_PI_8_BASIS = -1
ZERO_BASIS = 0
PI_8_BASIS = 1
PI_4_BASIS = 2

PROB_EQUAL = np.cos(np.pi/8) * np.cos(np.pi/8)
PROB_DIFFERENT = 1 - PROB_EQUAL

# Probability dictionary
dico = {
    (MINUS_PI_8_BASIS, ZERO_BASIS): [PROB_EQUAL, PROB_DIFFERENT],
    (MINUS_PI_8_BASIS, PI_8_BASIS): [0.5, 0.5],
    (MINUS_PI_8_BASIS, PI_4_BASIS): [PROB_DIFFERENT, PROB_EQUAL],
    (ZERO_BASIS, MINUS_PI_8_BASIS): [PROB_EQUAL, PROB_DIFFERENT],
    (ZERO_BASIS, PI_8_BASIS): [PROB_EQUAL, PROB_DIFFERENT],
    (ZERO_BASIS, PI_4_BASIS): [0.5, 0.5],
    (PI_8_BASIS, MINUS_PI_8_BASIS): [0.5, 0.5],
    (PI_8_BASIS, ZERO_BASIS): [PROB_EQUAL, PROB_DIFFERENT],
    (PI_8_BASIS, PI_4_BASIS): [PROB_EQUAL, PROB_DIFFERENT],
    (PI_4_BASIS, MINUS_PI_8_BASIS): [PROB_DIFFERENT, PROB_EQUAL],
    (PI_4_BASIS, ZERO_BASIS): [0.5, 0.5],
    (PI_4_BASIS, PI_8_BASIS): [PROB_EQUAL, PROB_DIFFERENT]
}

def measure_polarization(basis_a, basis_b):

    alice_bit = np.random.choice([0, 1], p=[0.5, 0.5])
    
    if basis_a == basis_b:
        bob_bit = alice_bit
    elif alice_bit == 0:
        bob_bit = np.random.choice([0, 1], p=dico[(basis_a, basis_b)])
    else:
        bob_bit = np.random.choice([1, 0], p=dico[(basis_a, basis_b)])

    return int(alice_bit), int(bob_bit)

def measure_polarization_eavesdropping(basis_a, basis_b, basis_e_a, basis_e_b):

    eve_alice_bit = np.random.choice([0, 1], p=[0.5, 0.5])
    
    if basis_e_a == basis_e_b:
        eve_bob_bit = eve_alice_bit
    elif eve_alice_bit == 0:
        eve_bob_bit = np.random.choice([0, 1], p=dico[(basis_e_a, basis_e_b)])
    else:
        eve_bob_bit = np.random.choice([1, 0], p=dico[(basis_e_a, basis_e_b)])

    if basis_e_a == basis_a:
        alice_bit = eve_alice_bit
    elif eve_alice_bit == 0:
        alice_bit = np.random.choice([0, 1], p=dico[(basis_a, basis_e_a)])
    else:
        alice_bit = np.random.choice([1, 0], p=dico[(basis_a, basis_e_a)])

    if basis_e_b == basis_b:
        bob_bit = eve_bob_bit
    elif eve_bob_bit == 0:
        bob_bit = np.random.choice([0, 1], p=dico[(basis_b, basis_e_b)])
    else:
        bob_bit = np.random.choice([1, 0], p=dico[(basis_b, basis_e_b)])

    return int(alice_bit), int(bob_bit), int(eve_alice_bit), int(eve_bob_bit)

def measure_polarization_eavesdropping_left(basis_a, basis_b, basis_e_a):

    eve_alice_bit = np.random.choice([0, 1], p=[0.5, 0.5])

    if basis_e_a == basis_a:
        alice_bit = eve_alice_bit
    elif eve_alice_bit == 0:
        alice_bit = np.random.choice([0, 1], p=dico[(basis_a, basis_e_a)])
    else:
        alice_bit = np.random.choice([1, 0], p=dico[(basis_a, basis_e_a)])

    if basis_e_a == basis_b:
        bob_bit = eve_alice_bit
    elif eve_alice_bit == 0:
        bob_bit = np.random.choice([0, 1], p=dico[(basis_b, basis_e_a)])
    else:
        bob_bit = np.random.choice([1, 0], p=dico[(basis_b, basis_e_a)])

    return int(alice_bit), int(bob_bit), int(eve_alice_bit)

def measure_polarization_eavesdropping_right(basis_a, basis_b, basis_e_b):

    eve_bob_bit = np.random.choice([0, 1], p=[0.5, 0.5])
    
    if basis_a == basis_e_b:
        alice_bit = eve_bob_bit
    elif eve_bob_bit == 0:
        alice_bit = np.random.choice([0, 1], p=dico[(basis_a, basis_e_b)])
    else:
        alice_bit = np.random.choice([1, 0], p=dico[(basis_a, basis_e_b)])

    if basis_b == basis_e_b:
        bob_bit = eve_bob_bit
    elif eve_bob_bit == 0:
        bob_bit = np.random.choice([0, 1], p=dico[(basis_e_b, basis_b)])
    else:
        bob_bit = np.random.choice([1, 0], p=dico[(basis_e_b, basis_b)])

    return int(alice_bit), int(bob_bit), int(eve_bob_bit)

def random_measure_polarization():

    # Alice's and Bob's random bases
    alice_basis = np.random.choice([ZERO_BASIS, PI_8_BASIS,PI_4_BASIS])
    bob_basis   = np.random.choice([MINUS_PI_8_BASIS, ZERO_BASIS, PI_8_BASIS])

    alice_bit, bob_bit = measure_polarization(alice_basis, bob_basis)

    return int(alice_basis), int(bob_basis), alice_bit, bob_bit