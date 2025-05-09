import os
import cv2
import numpy as np
import mediapipe as mp
from models import load_model
from utils import mediapipe_detection, extract_landmarks, prob_viz

def ensure_input_shape(sequence, target_shape=(20, 150)):
    """Ensure input matches target shape with efficient padding"""
    sequence = np.array(sequence)
    if len(sequence) > target_shape[0]:
        sequence = sequence[-target_shape[0]:]  # Use most recent frames
    elif len(sequence) < target_shape[0]:
        padding = np.zeros((target_shape[0] - len(sequence), target_shape[1]))
        sequence = np.concatenate([padding, sequence])
    return sequence[:, :target_shape[1]]  # Trim features if needed

def main():
    # Configuration
    MODEL_NAME = 'lstm_v3'
    DATA_PATH = 'keypoint_data'
    SEQUENCE_LENGTH = 20  # Fixed prediction window
    THRESHOLD = 0.80  # Confidence threshold
    
    # Initialize
    sequence = []
    predictions = []
    actions = sorted(os.listdir(DATA_PATH))
    last_prediction = None
    
    # Load model
    model = load_model(MODEL_NAME, len(actions), (SEQUENCE_LENGTH, 150))
    if model is None:
        print("Model failed to load")
        return

    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            image, results = mediapipe_detection(frame, holistic)
            landmarks = extract_landmarks(results)
            
            # Always add landmarks (but keep fixed sequence length)
            if len(sequence) < SEQUENCE_LENGTH:
                sequence.append(landmarks)
            else:
                sequence = sequence[1:] + [landmarks]  # Sliding window

            # Prediction trigger (EXACTLY at 20 frames)
            if len(sequence) == SEQUENCE_LENGTH:
                try:
                    processed_seq = ensure_input_shape(sequence)
                    res = model.predict(np.expand_dims(processed_seq, axis=0), verbose=0)[0]
                    
                    # Force display when buffer is full (even if low confidence)
                    max_conf = np.max(res)
                    action = actions[np.argmax(res)]
                    last_prediction = f"{action} ({max_conf*100:.0f}%)"
                    
                    # Store high-confidence predictions
                    if max_conf > THRESHOLD:
                        predictions.append(last_prediction)
                        if len(predictions) > 3:
                            predictions = predictions[-3:]
                
                except Exception as e:
                    print(f"Prediction error: {e}")

            # Display
            cv2.putText(image, f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Always show latest prediction (even if low confidence)
            if last_prediction:
                color = (0, 255, 0) if np.max(res) > THRESHOLD else (0, 0, 255)
                cv2.putText(image, last_prediction,
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Show prediction history
            cv2.putText(image, " | ".join(predictions),
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Control instructions
            cv2.putText(image, "ESC to exit | SPACE to reset",
                       (10, image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('ISL Translation', image)
            
            # Controls
            key = cv2.waitKey(10)
            if key in (27, 13):  # ESC/Enter
                break
            elif key == ord(' '):  # SPACE to reset
                sequence = []
                predictions = []
                last_prediction = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()