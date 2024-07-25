import numpy as np
from Pyfhel import Pyfhel

HE = Pyfhel()           # Creating empty Pyfhel object
n_mults = 10
ckks_params = {
    'scheme': 'CKKS',
    'n': 2**14,         # For CKKS, n/2 values can be encoded in a single ciphertext. 
    'scale': 2**31,     # Each multiplication grows the final scale
    'qi_sizes': [60]+ [30]*n_mults +[60] # Number of bits of each prime in the chain. 
                        # Intermediate prime sizes should be close to log2(scale).
                        # One per multiplication! More/higher qi_sizes means bigger 
                        #  ciphertexts and slower ops.
}
HE.contextGen(**ckks_params)  # Generate context for ckks scheme
HE.keyGen()             # Key Generation: generates a pair of public/secret keys
HE.relinKeyGen()




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

def multipleCiphers(x, y):
    res = x * y
    res = ~(res)
    return res












    