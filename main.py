import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from models import load_model
from utils import mediapipe_detection, extract_landmarks, prob_viz

def ensure_input_shape(sequence, target_shape=(20, 150)):
    sequence = np.array(sequence)
    if len(sequence) > target_shape[0]:
        sequence = sequence[-target_shape[0]:]
    elif len(sequence) < target_shape[0]:
        padding = np.zeros((target_shape[0] - len(sequence), target_shape[1]))
        sequence = np.concatenate([padding, sequence])
    return sequence[:, :target_shape[1]]

def draw_odia_text(image, text, position, font_size=24, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("NotoSansOriya-Regular.ttf", font_size)
    except:
        font = ImageFont.load_default()
        print("Warning: Odia font not found, using default font")
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    MODEL_NAME = 'lstm_v3'
    DATA_PATH = 'keypoint_data'
    SEQUENCE_LENGTH = 20
    THRESHOLD = 0.80

    odia_translations = {
        "Hello": "ନମସ୍କାର",
        "How are you": "ଆପଣ କେମିତି ଅଛନ୍ତି?",
        "Alright": "ଭଲ ଅଛି",
        "Good Morning": "ଶୁଭ ସକାଳ",
        "Good afternoon": "ଶୁଭ ଅପରାହ୍ନ",
        "Good evening": "ଶୁଭ ସନ୍ଧ୍ୟା",
        "Good night": "ଶୁଭରାତ୍ରି",
        "Thank you": "ଧନ୍ୟବାଦ",
        "Pleased": "ଆନନ୍ଦିତ"
    }

    sequence = []
    predictions = []
    actions = sorted([folder.strip() for folder in os.listdir(DATA_PATH)])
    last_prediction = None

    model = load_model(MODEL_NAME, len(actions), (SEQUENCE_LENGTH, 150))
    if model is None:
        print("Model failed to load")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            landmarks = extract_landmarks(results)

            if len(sequence) < SEQUENCE_LENGTH:
                sequence.append(landmarks)
            else:
                sequence = sequence[1:] + [landmarks]

            if len(sequence) == SEQUENCE_LENGTH:
                try:
                    processed_seq = ensure_input_shape(sequence)
                    res = model.predict(np.expand_dims(processed_seq, axis=0), verbose=0)[0]
                    max_conf = np.max(res)
                    action = actions[np.argmax(res)]
                    odia_action = odia_translations.get(action, action)
                    last_prediction = f"{odia_action} ({max_conf*100:.0f}%)"

                    if max_conf > THRESHOLD:
                        predictions.append(last_prediction)
                        if len(predictions) > 3:
                            predictions = predictions[-3:]
                except Exception as e:
                    print(f"Prediction error: {e}")

            image = draw_odia_text(image, f"ଫ୍ରେମ୍: {len(sequence)}/{SEQUENCE_LENGTH}", (10, 20), 22, (255, 255, 0))

            if last_prediction:
                color = (0, 255, 0) if np.max(res) > THRESHOLD else (0, 0, 255)
                image = draw_odia_text(image, last_prediction, (10, 60), 26, color)

            if predictions:
                history_text = " | ".join(predictions)
                image = draw_odia_text(image, history_text, (10, 100), 20, (0, 255, 255))

            image = draw_odia_text(image, "ESC ଦବାନ୍ତୁ ବାହାରିବାକୁ | SPACE ଦବାନ୍ତୁ ରିସେଟ୍ କରିବାକୁ",
                                   (10, image.shape[0] - 30), 20, (255, 255, 255))

            cv2.imshow('ISL ଅନୁବାଦକ', image)

            key = cv2.waitKey(10)
            if key in (27, 13):
                break
            elif key == ord(' '):
                sequence = []
                predictions = []
                last_prediction = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
