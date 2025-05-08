import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

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

def augment_data(sequences, labels):
    """Augment the dataset with noise and time warping"""
    augmented_seqs = []
    augmented_labels = []
    
    for seq, label in zip(sequences, labels):
        # Add original
        augmented_seqs.append(seq)
        augmented_labels.append(label)
        
        # Add noise (only to training data)
        noisy_seq = seq + np.random.normal(0, 0.005, seq.shape)  # Reduced noise
        augmented_seqs.append(noisy_seq)
        augmented_labels.append(label)
        
        # Time warping (if temporal data)
        if len(seq) > 10:
            warp_factor = np.random.uniform(0.95, 1.05)  # Reduced warping
            warped_seq = tf.image.resize(seq[np.newaxis, ..., np.newaxis], 
                                       [int(30*warp_factor), 150])
            warped_seq = tf.image.resize(warped_seq, [30, 150])
            augmented_seqs.append(warped_seq[0, ..., 0].numpy())
            augmented_labels.append(label)
    
    return np.array(augmented_seqs), np.array(augmented_labels)

def create_improved_model(input_shape, num_classes):
    """Create an improved bidirectional LSTM model with regularization"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        
        # First LSTM layer with dropout
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.BatchNormalization(),
        
        # Second LSTM layer
        tf.keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.BatchNormalization(),
        
        # Dense layers
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def compile_improved_model(model):
    """Compile model with enhanced optimizer and metrics"""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005,  # Reduced learning rate
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )
    return model

def get_enhanced_callbacks(model_dir):
    """Create enhanced callbacks for training"""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, "best_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, "logs"),
            histogram_freq=1
        )
    ]

def evaluate_model(model, X_test, y_test, actions):
    """Evaluate model performance with comprehensive metrics"""
    print("\nFinal Evaluation:")
    results = model.evaluate(X_test, y_test, verbose=0)
    
    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'precision': results[2],
        'recall': results[3],
        'auc': results[4]
    }
    
    print(f"\nTest Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test AUC: {metrics['auc']:.4f}")
    
    # Confusion matrix
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    cm = tf.math.confusion_matrix(y_true, y_pred_classes).numpy()
    print("\nConfusion Matrix:")
    print(cm)
    
    # Class-wise metrics
    print("\nClass-wise Performance:")
    for i, action in enumerate(actions):
        tp = cm[i,i]
        fp = cm[:,i].sum() - tp
        fn = cm[i,:].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"{action}: Precision={precision:.2f}, Recall={recall:.2f}")

def train_improved():
    """Main training function with all improvements"""
    # Configuration
    DATA_PATH = 'keypoint_data'
    MODEL_NAME = 'lstm_improved_v2'
    INPUT_SHAPE = (30, 150)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and validate data
    X, y, actions = load_and_validate_data(DATA_PATH)
    print(f"Loaded dataset with {len(actions)} actions")
    print(f"Input shape: {X.shape}, Labels shape: {y.shape}")
    
    # Train-test split (before augmentation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Data augmentation (only on training data)
    print("\nAugmenting training data...")
    X_train, y_train = augment_data(X_train, y_train)
    print(f"Augmented training data shape: {X_train.shape}")
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")
    
    # Model setup
    print("\nCreating model...")
    model = create_improved_model(INPUT_SHAPE, len(actions))
    model = compile_improved_model(model)
    model.summary()
    
    # Callbacks
    model_dir = os.path.join("models", MODEL_NAME)
    os.makedirs(model_dir, exist_ok=True)
    callbacks = get_enhanced_callbacks(model_dir)
    
    # Training
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,  # Reduced epochs
        batch_size=16,  # Smaller batch size
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluation
    evaluate_model(model, X_test, y_test, actions)
    
    # Save final model
    model.save(os.path.join(model_dir, "final_model.keras"))
    print(f"\nTraining complete. Model saved to {model_dir}")

if __name__ == "__main__":
    # Configure TensorFlow for better performance
    tf.config.optimizer.set_jit(True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    
    train_improved()