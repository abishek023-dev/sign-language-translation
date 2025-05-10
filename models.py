import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input
import os

def build_model(input_shape=(30, 165), num_classes=0):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def load_model(model_name, input_shape=(30, 165), num_classes=0):
    model_path = f"{model_name}.keras"
    if os.path.exists(model_path):
        print(f"[INFO] Loaded existing model from {model_path}")
        return tf.keras.models.load_model(model_path)
    else:
        print(f"[INFO] No existing model found. Creating new model...")
        return build_model(input_shape, num_classes)
