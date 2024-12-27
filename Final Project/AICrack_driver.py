from aicrack_train import train_model
from aicrack_utils import encode_data, binary_to_hex, hex_to_binary
import joblib
import numpy as np
import random
import string
import binascii


def encrypt(plaintext, key):
    plainhex = binascii.hexlify(plaintext.encode())
    plainbin = bin(int(plainhex, 16))[2:].zfill(8 * ((len(plainhex) + 1) // 2))
    keyhex = binascii.hexlify(key.encode())
    keybin = bin(int(keyhex, 16))[2:].zfill(8 * ((len(keyhex) + 1) // 2))
    
    lefttext = plainbin[:32]
    righttext = plainbin[32:]
    keyleft = keybin[:32]
    keyright = keybin[32:]
    
    # Two rounds of encryption
    for i in range(2):
        if i % 2 != 0:
            lefttext, righttext, keyleft, keyright = leftroundencrypt(lefttext, righttext, keyleft, keyright)
        else:
            lefttext, righttext, keyleft, keyright = rightroundencrypt(lefttext, righttext, keyleft, keyright)

    return lefttext + righttext  # Return ciphertext in binary form


def leftroundencrypt(plainl, plainr, keyl, keyr):
    right = ""
    for i in range(len(plainr)):
        right = right + str(int(plainr[i]) ^ int(keyl[i]))
    out = ""
    for i in range(len(right)):
        out = out + str(int(right[i]) ^ int(plainl[i]))
    keyl = keyl[4:] + keyl[:4]
    return plainr, out, keyl, keyr


def rightroundencrypt(plainl, plainr, keyl, keyr):
    right = ""
    for i in range(len(plainr)):
        right = right + str(int(plainr[i]) ^ int(keyr[i]))
    out = ""
    for i in range(len(right)):
        out = out + str(int(right[i]) ^ int(plainl[i]))
    keyr = keyr[4:] + keyr[:4]
    return plainr, out, keyl, keyr

# Function to generate random strings of a given length
def generate_random_string(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

# Function to generate encrypted ciphertext from plaintext using your Feistel encryption
def generate_encrypted_data(n_samples, block_size, key):
    plaintexts = []
    ciphertexts = []

    for _ in range(n_samples):
        plaintext = generate_random_string(block_size // 8)  # Generate random plaintext (8-bit per character)
        ciphertext = encrypt(plaintext, key)
        plaintexts.append(plaintext)
        ciphertexts.append(binary_to_hex(ciphertext))  # Store ciphertext as hex for comparison

    return plaintexts, ciphertexts

# Function to evaluate how well the trained model performs on generated test data
def evaluate_model(model, n_samples, block_size, key):
    print("Generating test data...")
    plaintexts, ciphertexts = generate_encrypted_data(n_samples, block_size, key)

    # Prepare the data for prediction
    inputs = []
    outputs = []
    for pt, ct in zip(plaintexts, ciphertexts):
        pt_bin = ''.join(format(ord(char), '08b') for char in pt).zfill(block_size)  # Convert plaintext to binary
        ct_bin = hex_to_binary(ct).zfill(block_size)  # Convert ciphertext from hex to binary
        inputs.append([int(bit) for bit in pt_bin])
        outputs.append([int(bit) for bit in ct_bin])

    X = np.array(inputs)
    y = np.array(outputs)

    # Predict using the trained model
    predictions = model.predict(X)

    # Compare the predicted ciphertext with the true ciphertext
    correct_predictions = 0
    total_predictions = len(predictions)

    for i in range(total_predictions):
        if np.array_equal(predictions[i], y[i]):
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Model Accuracy on Test Data: {accuracy:.2f}%")
    
    # Show sample comparisons
    print("Sample Results:")
    for i in range(min(5, total_predictions)):  # Show a few samples
        print(f"Sample {i + 1}:")
        print(f"Plaintext: {plaintexts[i]}")
        print(f"True Ciphertext: {ciphertexts[i]}")
        
        # Convert prediction (array of 0s and 1s) to a binary string
        predicted_bin_str = ''.join(map(str, predictions[i]))

        # Print predicted ciphertext after converting binary string to hex
        print(f"Predicted Ciphertext: {binary_to_hex(predicted_bin_str)}")
        print()



def main():
    model_path = "feistel_model.pkl"
    block_size = 64  # 64-bit block size (adjust as needed)
    key = "ABCD1234"  # Example key (can be any string)

    print("Loading the trained model...")
    model = joblib.load(model_path)

    n_samples = 5000  # Number of test samples

    print("Evaluating the trained model on generated test data...")
    evaluate_model(model, n_samples, block_size, key)


if __name__ == "__main__":
    main()
