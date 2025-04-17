import os
import cv2
import numpy as np
import mediapipe as mp
from models import load_model
from utils import mediapipe_detection, extract_landmarks, prob_viz

def ensure_input_shape(sequence, target_shape=(30, 150)):
    """Ensures input matches target shape of (30, 150)"""
    sequence = np.array(sequence)
    
    # Pad or trim frames
    if sequence.shape[0] > target_shape[0]:
        sequence = sequence[:target_shape[0]]
    elif sequence.shape[0] < target_shape[0]:
        padding = np.zeros((target_shape[0] - sequence.shape[0], target_shape[1]))
        sequence = np.concatenate([sequence, padding])
    
    # Ensure 150 features
    if sequence.shape[1] > target_shape[1]:
        sequence = sequence[:, :target_shape[1]]
    elif sequence.shape[1] < target_shape[1]:
        padding = np.zeros((sequence.shape[0], target_shape[1] - sequence.shape[1]))
        sequence = np.concatenate([sequence, padding], axis=1)
    
    return sequence

def main():
    # Configuration
    MODEL_NAME = 'lstm_v3'
    DATA_PATH = 'keypoint_data'
    SEQUENCE_LENGTH = 30
    THRESHOLD = 0.8
    
    # Initialize
    sequence = []
    predictions = []
    actions = sorted(os.listdir(DATA_PATH))
    
    # Load model
    model = load_model(MODEL_NAME, len(actions), (SEQUENCE_LENGTH, 150))
    if model is None:
        print("Failed to initialize model")
        return
    
    # Camera setup
    cap = cv2.VideoCapture(0)
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            image, results = mediapipe_detection(frame, holistic)
            landmarks = extract_landmarks(results)
            sequence.append(landmarks)
            
            # Make prediction when buffer is full
            if len(sequence) == SEQUENCE_LENGTH:
                try:
                    processed_seq = ensure_input_shape(sequence)
                    res = model.predict(np.expand_dims(processed_seq, axis=0), verbose=0)[0]
                    sequence = []  # Reset buffer
                    
                    if np.max(res) > THRESHOLD:
                        action = actions[np.argmax(res)]
                        predictions.append(action)
                        if len(predictions) > 5:
                            predictions = predictions[-5:]
                except Exception as e:
                    print(f"Prediction error: {e}")
                    sequence = []
            
            # Display results
            image = prob_viz(res, actions, image) if 'res' in locals() else image
            cv2.putText(image, ' '.join(predictions), 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            cv2.imshow('ISL Translation', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()