import numpy as np
from Pyfhel import Pyfhel
import concurrent.futures

# def initialize_HE():
#     n_mults = 10
#     ckks_params = {
#         'scheme': 'CKKS',
#         'n': 2**14,
#         'scale': 2**31,
#         'qi_sizes': [60] + [30]*n_mults + [60]
#     }
#     HE = Pyfhel()
#     HE.contextGen(**ckks_params)
#     HE.keyGen()
#     HE.relinKeyGen()
#     return HE

# HE = initialize_HE()

def encryptMatrix(matrix, HE):
    matrix = np.array(matrix)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        encrypted_matrix = list(executor.map(HE.encrypt, matrix.flat))
    return np.array(encrypted_matrix).reshape(matrix.shape)

def decryptMatrix(encrypted_matrix, HE):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        decrypted_matrix = list(executor.map(lambda x: HE.decrypt(x)[0], encrypted_matrix.flat))
    return np.array(decrypted_matrix).reshape(encrypted_matrix.shape)

def multipleCiphers(x, y):
    res = x * y
    res = ~(res)
    return res

# if __name__ == '__main__':
#     mat1 = np.array([[1, 2], [3, 4]])
#     mat2 = np.array([[5, 6], [7, 8]])

#     eMat1 = encryptMatrix(mat1)
#     eMat2 = encryptMatrix(mat2)

#     # Perform dot product on encrypted matrices
#     res = np.dot(eMat1, eMat2)

#     dRes = decryptMatrix(res)

#     print("Decrypted result:")
#     print(dRes)
#     print("\nExpected result:")
#     print(np.dot(mat1, mat2))