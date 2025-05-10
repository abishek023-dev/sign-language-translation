import numpy as np
import tensorflow as tf
from models import load_model
from utils import get_data, accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Load test data
X_train, y_train = get_data(train=True, test=False)
X_test, y_test = get_data(train=False, test=True)

# Input shape: (sequence_length, feature_dim) = (30, 165)
sequence_length = X_train.shape[1]
feature_dim = X_train.shape[2]
num_classes = y_train.shape[1]

# Load model (create if doesn't exist)
model = load_model('lstm_v3', input_shape=(sequence_length, feature_dim), num_classes=num_classes)

# Callbacks
checkpoint = ModelCheckpoint("isl_model.keras", save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Training
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=100,
          batch_size=16,
          callbacks=[checkpoint, reduce_lr, early_stop])

# Final accuracy
final_acc = accuracy(model, X_test, y_test)
print(f"Test Accuracy: {final_acc * 100:.2f}%")
