import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from models import load_model

# Setup device
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

# Load dataset
IMPORT_PATH = 'keypoint_data'
actions = sorted(os.listdir(IMPORT_PATH))
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for file in os.listdir(os.path.join(IMPORT_PATH, action)):
        filepath = os.path.join(IMPORT_PATH, action, file)
        sequences.append(np.load(filepath))
        labels.append(label_map[action])

X = np.array(sequences, dtype=np.float32)
y = to_categorical(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# Load model
model = load_model('lstm_v3', pretrained=False)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("isl_model.h5", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, cooldown=15, min_delta=0.0001),
]

# Train
with tf.device(device):
    model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test, y_test), callbacks=callbacks)
