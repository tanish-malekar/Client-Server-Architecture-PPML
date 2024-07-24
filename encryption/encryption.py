import numpy as np
from Pyfhel import Pyfhel

HE = Pyfhel()           # Creating empty Pyfhel object
ckks_params = {
    'scheme': 'CKKS',   # can also be 'ckks'
    'n': 2**14,         # Polynomial modulus degree. For CKKS, n/2 values can be
                        #  encoded in a single ciphertext.
                        #  Typ. 2^D for D in [10, 15]
    'scale': 2**30,     # All the encodings will use it for float->fixed point
                        #  conversion: x_fix = round(x_float * scale)
                        #  You can use this as default scale or use a different
                        #  scale on each operation (set in HE.encryptFrac)
    'qi_sizes': [60, 30, 30, 30, 60] # Number of bits of each prime in the chain.
                        # Intermediate values should be  close to log2(scale)
                        # for each operation, to have small rounding errors.
}
HE.contextGen(**ckks_params)  # Generate context for ckks scheme
HE.keyGen()             # Key Generation: generates a pair of public/secret keys
HE.rotateKeyGen()




def encryptMatrix(matrix, HE):
    encrypted_matrix = []
    for row in matrix:
        encrypted_row = [HE.encrypt(val) for val in row]
        encrypted_matrix.append(encrypted_row)
    return encrypted_matrix

def decryptMatrix(encrypted_matrix, HE):
    decrypted_matrix = []
    for row in encrypted_matrix:
        decrypted_row = [HE.decrypt(val)[0] for val in row]
        decrypted_matrix.append(decrypted_row)
    return decrypted_matrix











    