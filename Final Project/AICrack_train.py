# aicrack_train.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
from aicrack_utils import hex_to_binary, binary_to_hex


def read_file(file_path):
    """
    Reads data from a file.
    Each line should contain one entry.
    """
    with open(file_path, "r") as file:
        return [line.strip() for line in file]


def prepare_data(plaintext_file, ciphertext_file, block_size):
    """
    Reads plaintext and ciphertext pairs from files
    and prepares the data for training.
    """
    print("Loading data...")
    plaintexts = read_file(plaintext_file)
    ciphertexts = read_file(ciphertext_file)

    assert len(plaintexts) == len(ciphertexts), "Mismatch in number of plaintext and ciphertext entries."

    print("Encoding data...")
    inputs = []
    outputs = []

    for pt, ct in zip(plaintexts, ciphertexts):
        # Convert plaintext (string) to binary representation
        pt_bin = ''.join(f"{ord(char):08b}" for char in pt).zfill(block_size)
        # Convert ciphertext (hex) to binary representation
        ct_bin = hex_to_binary(ct).zfill(block_size)
        inputs.append([int(bit) for bit in pt_bin])
        outputs.append([int(bit) for bit in ct_bin])

    return np.array(inputs), np.array(outputs)


def train_model(plaintext_file, ciphertext_file, block_size, model_path):
    """
    Train a neural network to approximate a Feistel cipher's input-output mapping.
    """
    X, y = prepare_data(plaintext_file, ciphertext_file, block_size)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training the model...")
    model = MLPClassifier(hidden_layer_sizes=(512, 128, 64), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating the model...")
    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)

    return model


if __name__ == "__main__":
    plaintext_file = "Cryptography456HW/plaintext.txt"
    ciphertext_file = "Cryptography456HW/ciphertext.txt"
    model_path = "feistel_model.pkl"
    block_size = 64  # 64-bit block size

    print("Starting training...")
    train_model(plaintext_file, ciphertext_file, block_size, model_path)
