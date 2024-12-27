# aicrack_utils.py
import numpy as np
import random
import binascii
def generate_feistel_data(n_samples, block_size, rounds, key, round_function):
    """
    Generate plaintext-ciphertext pairs for a Feistel cipher.
    """
    plaintexts = [random.randint(0, (1 << block_size) - 1) for _ in range(n_samples)]
    pairs = []

    for pt in plaintexts:
        left = pt >> (block_size // 2)
        right = pt & ((1 << (block_size // 2)) - 1)

        for _ in range(rounds):
            new_left = right
            new_right = left ^ round_function(right, key)
            left, right = new_left, new_right

        ciphertext = (left << (block_size // 2)) | right
        pairs.append((pt, ciphertext))

    return pairs

def simple_round_function(right, key):
    """
    Example round function: XOR with key and rotate right bits.
    """
    return ((right ^ key) >> 3) | ((right ^ key) << (13))

def encode_data(pairs, block_size):
    """
    Convert plaintext-ciphertext pairs into input-output vectors for ML training.
    """
    inputs = []
    outputs = []

    for pt, ct in pairs:
        pt_bin = np.unpackbits(np.array([pt], dtype=np.uint16).view(np.uint8))
        ct_bin = np.unpackbits(np.array([ct], dtype=np.uint16).view(np.uint8))
        inputs.append(pt_bin)
        outputs.append(ct_bin)

    return np.array(inputs), np.array(outputs)

def hex_to_binary(hex):
    # Convert hexadecimal string to binary string
    binary_str = bin(int(hex, 16))[2:]  # [2:] removes the '0b' prefix
    return binary_str


def hex_to_binary(hex_str, bit_length=None):
    # Ensure even length by padding with leading zero if necessary
    if len(hex_str) % 2 != 0:
        hex_str = '0' + hex_str

    # Convert hex to binary
    bytes_data = binascii.unhexlify(hex_str)
    return ''.join(f"{byte:08b}" for byte in bytes_data)

def binary_to_hex(bin_str, hex_length=None):
    # Convert binary string to bytes
    byte_data = int(bin_str, 2).to_bytes((len(bin_str) + 7) // 8, byteorder='big')
    # Convert bytes to hex string
    hex_str = binascii.hexlify(byte_data).decode().upper()  # Convert to uppercase for consistency
    if hex_length:
        hex_str = hex_str.zfill(hex_length)
    return hex_str