import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from utils import mediapipe_detection, draw_styled_landmarks, landmarks_data, prob_viz
from PIL import Image, ImageDraw, ImageFont  # For Odia text rendering
import os

# Odia actions
actions = ["ଠିକ୍ ଅଛି", "ଶୁଭ ଅପରାହ୍ନ", "ଶୁଭ ସନ୍ଧ୍ୟା", "ଶୁଭ ପ୍ରଭାତ", 
           "ଶୁଭ ରାତ୍ରି", "ନମସ୍କାର", "କେମିତି ଅଛନ୍ତି?", "ଖୁସି ହେଲି", "ଧନ୍ୟବାଦ"]

sequence = []
sentence = []
threshold = 0.1

# Load Odia font
font_path = "NotoSansOriya-Regular.ttf"  # Make sure this file is in the same directory
if os.path.exists(font_path):
    font = ImageFont.truetype(font_path, 30)
else:
    print("⚠️ Odia font not found! Falling back to default font.")
    font = ImageFont.load_default()

# Define the model
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Load or create the model
model = create_model(input_shape=(30, 150), num_classes=len(actions))
try:
    model.load_weights('isl_model.keras')
    print("✅ Model weights loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model weights: {e}")

# Mediapipe setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mediapipe detection
    image, results = mediapipe_detection(frame, holistic)
    draw_styled_landmarks(image, results)

    # Extract keypoints
    keypoints = landmarks_data(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        input_data = np.expand_dims(sequence, axis=0)
        res = model.predict(input_data)[0]

        if res[np.argmax(res)] > threshold:
            action = actions[np.argmax(res)]
            if len(sentence) == 0 or action != sentence[-1]:
                sentence.append(action)

        if len(sentence) > 5:
            sentence = sentence[-5:]

        # Display probabilities
        image = prob_viz(res, actions, image)

    # Convert to PIL for Odia text rendering
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Draw header background
    header_height = 60
    draw.rectangle([(0, 0), (image.shape[1], header_height)], fill=(50, 50, 150))

    # Draw Odia text sentence
    text_y_position = 15
    draw.text((10, text_y_position), " ".join(sentence), font=font, fill=(255, 255, 0))

    # Convert back to OpenCV format
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Display frame
    cv2.imshow('Odia Sign Language Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
