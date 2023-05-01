import numpy as np
sequence = ['A', 'C', 'T', 'U', 'G', 'N']

def one_hot_encode(sequence):
    encoding_dict = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'U': [0, 0, 0, 1],
        'N': [0, 0, 0, 0],
    }

    encoded_sequence = [encoding_dict[base] for base in sequence]
    return np.array(encoded_sequence)

one_hot_encoded_sequence = one_hot_encode(sequence)
print(one_hot_encoded_sequence)