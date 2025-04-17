import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from models import create_model, compile_model

def load_and_validate_data(data_path):
    """Loads and validates dataset with consistent 150 features"""
    actions = sorted(os.listdir(data_path))
    sequences = []
    labels = []
    
    for action_idx, action in enumerate(actions):
        action_dir = os.path.join(data_path, action)
        for seq_file in os.listdir(action_dir):
            seq_path = os.path.join(action_dir, seq_file)
            seq_data = np.load(seq_path)
            
            # Validate shape
            if seq_data.shape != (30, 150):
                print(f"Warning: File {seq_path} has shape {seq_data.shape}, resizing to (30, 150)")
                if seq_data.shape[1] > 150:
                    seq_data = seq_data[:, :150]
                else:
                    pad_width = ((0, 0), (0, 150 - seq_data.shape[1]))
                    seq_data = np.pad(seq_data, pad_width)
            
            sequences.append(seq_data)
            labels.append(action_idx)
    
    return np.array(sequences), to_categorical(labels), actions

def train():
    # Configuration
    DATA_PATH = 'keypoint_data'
    MODEL_NAME = 'lstm_v3'
    INPUT_SHAPE = (30, 150)
    
    # Load and validate data
    X, y, actions = load_and_validate_data(DATA_PATH)
    print(f"Loaded dataset with {len(actions)} actions")
    print(f"Input shape: {X.shape}, Labels shape: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Model setup
    model = create_model(INPUT_SHAPE, len(actions))
    model = compile_model(model)
    
    # Callbacks
    model_dir = os.path.join("models", MODEL_NAME)
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, "model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
    ]
    
    # Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )
    
    print(f"Training complete. Model saved to {model_dir}")

if __name__ == "__main__":
    train()