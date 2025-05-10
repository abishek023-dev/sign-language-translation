import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from utils import mediapipe_detection, draw_styled_landmarks, landmarks_data, prob_viz  # Assuming these are in a utils.py file
import os
# Constants
keypoint_data_dir = 'keypoint_data'  # Path to the directory containing keypoint data
actions = sorted(os.listdir(keypoint_data_dir))  # This will list all subdirectories (actions) alphabetically
print(f"Actions list: {actions}")  # This will print out the actions to verify
sequence = []
sentence = []
threshold = 0.7

# Define the model (Updated to match input data shape)
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))  # Adjusted to accept input_shape
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Load or create the model
model = create_model(input_shape=(30, 150), num_classes=len(actions))  # Adjusted to 150 features per frame
try:
    model.load_weights('isl_model.keras')  # Load weights from the saved model
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")

# Mediapipe setup for hand/pose tracking
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# OpenCV video capture setup
cap = cv2.VideoCapture(0)

# Main loop for real-time video processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mediapipe detection
    image, results = mediapipe_detection(frame, holistic)
    draw_styled_landmarks(image, results)

    # Extract keypoints from detected results
    keypoints = landmarks_data(results)  # Assuming this function returns a 150-dim feature vector for each frame
    print(f"Keypoints Shape: {np.array(keypoints).shape}")  # Check the keypoints shape to ensure it's (150,)
    sequence.append(keypoints)
    sequence = sequence[-30:]  # Keep last 30 frames

    if len(sequence) == 30:
        # Prepare the data for prediction
        input_data = np.expand_dims(sequence, axis=0)  # Shape should be (1, 30, 150)
        print(f"Input Data Shape: {input_data.shape}")  # Check input shape

        # Model prediction
        res = model.predict(input_data)[0]

        if res[np.argmax(res)] > threshold:
            action = actions[np.argmax(res)]
            if len(sentence) == 0 or action != sentence[-1]:
                sentence.append(action)

        if len(sentence) > 5:  # Keep the last 5 actions
            sentence = sentence[-5:]

        # Display the predicted actions on the frame
        image = prob_viz(res, actions, image)

    # Show predicted sentence on the frame
    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, ' '.join(sentence), (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('OpenCV Feed', image)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
