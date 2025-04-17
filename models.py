import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as tf_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_model(input_shape, num_classes):
    """Creates LSTM model with correct input dimensions"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def compile_model(model):
    """Compiles the model with appropriate settings"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_model(model_name, num_classes, input_shape):
    """
    Loads or creates model with proper error handling
    """
    model_path = os.path.join("models", model_name, "model.keras")
    
    if os.path.exists(model_path):
        try:
            model = tf_load_model(model_path)
            print(f"Loaded existing model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model instead")
    
    # Create new model if loading fails
    model = create_model(input_shape, num_classes)
    model = compile_model(model)
    return model