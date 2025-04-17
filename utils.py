import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    """Process image with MediaPipe Holistic"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def extract_landmarks(results):
    """Extracts and formats landmarks to consistent 150 features"""
    # Pose landmarks (33 points, 3 values each)
    pose = np.array([[res.x, res.y, res.z] for res in 
                    results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    
    # Left hand landmarks (21 points)
    lh = np.array([[res.x, res.y, res.z] for res in 
                  results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Right hand landmarks (21 points)
    rh = np.array([[res.x, res.y, res.z] for res in 
                  results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Combine and ensure 150 features
    features = np.concatenate([pose, lh, rh])
    if len(features) > 150:
        return features[:150]
    elif len(features) < 150:
        return np.pad(features, (0, 150 - len(features)))
    return features

def prob_viz(res, actions, image):
    """Visualizes prediction probabilities"""
    output = image.copy()
    bar_width = 200
    start_y = 30
    
    if res is None or len(actions) == 0:
        return output
    
    for i, (action, prob) in enumerate(zip(actions, res)):
        if i >= 5:  # Only show top 5 predictions
            break
        cv2.rectangle(output, (10, start_y + i*30), 
                     (10 + int(prob * bar_width), start_y + 20 + i*30),
                     (0, 255, 0), -1)
        cv2.putText(output, f"{action}: {prob:.2f}", 
                   (15, start_y + 15 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    return output