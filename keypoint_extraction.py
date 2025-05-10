import os
import cv2
import mediapipe as mp
import numpy as np
from utils import mediapipe_detection, landmarks_data, pad_sequence
from tqdm import tqdm
import re

def clean_action_name(action_folder):
    """
    Extracts the meaningful action name (ignores numbers, trims, lowercases)
    e.g., '48. Hello' -> 'hello'
    """
    return re.sub(r"^\d+\.\s*", "", action_folder).strip().lower()

def save_data(action, video_file, export_path, import_path, max_frame_length=30, skip_frame=2):
    frame_count = 0
    processed = 0
    data_per_video = []

    video_path = os.path.join(import_path, action, video_file)
    cap = cv2.VideoCapture(video_path)

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1

            if not ret:
                break

            if frame_count % skip_frame == 0:
                image, results = mediapipe_detection(frame, holistic)
                keypoints = landmarks_data(results)
                
                if keypoints is not None:
                    data_per_video.append(keypoints)
                    processed += 1

            if processed == max_frame_length:
                break

    cap.release()

    # Pad if shorter than expected
    if processed < max_frame_length:
        data_per_video = pad_sequence(data_per_video, max_frame_length)
    data_per_video = np.array(data_per_video)

    cleaned_action = clean_action_name(action)
    os.makedirs(os.path.join(export_path, cleaned_action), exist_ok=True)
    npy_path = os.path.join(export_path, cleaned_action, f"{os.path.splitext(video_file)[0]}_skip_{skip_frame}.npy")
    
    print(f"[✔] Saved {cleaned_action}/{video_file} → Shape: {data_per_video.shape}")
    np.save(npy_path, data_per_video)


if __name__ == "__main__":
    EXPORT_PATH = 'keypoint_data'
    IMPORT_PATH = 'greetings_data'
    MAX_FRAME_LENGTH = 30
    SKIP_FRAME = 2

    video_exts = ('.mp4', '.avi', '.mov')

    for action in tqdm(os.listdir(IMPORT_PATH), desc="Processing folders"):
        action_path = os.path.join(IMPORT_PATH, action)
        if not os.path.isdir(action_path):
            continue

        for video_file in os.listdir(action_path):
            if video_file.lower().endswith(video_exts):
                try:
                    save_data(action, video_file, EXPORT_PATH, IMPORT_PATH, MAX_FRAME_LENGTH, SKIP_FRAME)
                except Exception as e:
                    print(f"[ERROR] Failed on {action}/{video_file} → {str(e)}")
