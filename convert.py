import tensorflow as tf

old_model_path = "models/lstm_v3/model.h5"  # Your old model file
new_model_path = "models/lstm_v3/model.keras"  # New model file

# Load the old model
model = tf.keras.models.load_model(old_model_path, safe_mode=False)

# Save it in new format
model.save(new_model_path)

print(f"✅ Model converted and saved as: {new_model_path}")
