# AICrack_train.py
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from AICrack_utils import load_data, encode_data

def prepare_labels(y_train_hex):
    # Prepare label encoders for each character in the 16-character hex output
    y_train_split = np.array([list(y) for y in y_train_hex])  # Split each hex string into characters
    label_encoders = [LabelEncoder() for _ in range(16)]
    y_train_class = np.zeros(y_train_split.shape, dtype=int)

    for i in range(16):
        y_train_class[:, i] = label_encoders[i].fit_transform(y_train_split[:, i])

    return y_train_class, label_encoders

def train_model(x_train, y_train_class):
    # Train a separate classifier for each of the 16 hex characters
    models = []
    for i in range(16):
        model = MLPClassifier(
            hidden_layer_sizes=(64,),  # Fewer neurons and layers to reduce complexity
            activation='relu',
            max_iter=1200,
            solver='adam',
            learning_rate_init=0.0001,
            alpha=0.1
        )
        model.fit(x_train, y_train_class[:, i])
        models.append(model)
    return models

def main():
    print("Loading data...")
    plaintext, ciphertext = load_data()

    print("Encoding X...")
    x_train = encode_data(plaintext, max_len=8)

    print("Preparing Y as classification targets...")
    y_train_class, label_encoders = prepare_labels(ciphertext)

    print("Training Models...")
    models = train_model(x_train, y_train_class)

    print("Saving Models and Encoders...")
    # Save each character classifier and label encoders using joblib
    joblib.dump(models, 'AICrack_models.pkl')
    joblib.dump(label_encoders, 'AICrack_label_encoders.pkl')

if __name__ == '__main__':
    main()
