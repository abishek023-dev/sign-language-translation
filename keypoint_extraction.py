# keypoint_extraction.py
import os
import cv2
import mediapipe as mp
import numpy as np
from utils import mediapipe_detection, landmarks_data, pad_sequence

def save_data(action, video_file, export_path, import_path, max_frame_length=30, skip_frame=2):
    frame_count = 0
    processed = 0
    data_per_video = []

    cap = cv2.VideoCapture(os.path.join(import_path, action, video_file))
    
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1

            if ret and (frame_count % skip_frame == 0):
                image, results = mediapipe_detection(frame, holistic)
                processed += 1
                
                keypoints = landmarks_data(results)
                if keypoints is not None:
                    data_per_video.append(keypoints)

            if not ret or processed == max_frame_length:
                if processed != max_frame_length:
                    data_per_video = np.array(pad_sequence(data_per_video, max_frame_length))
                else:
                    data_per_video = np.array(data_per_video)
                
                os.makedirs(os.path.join(export_path, action), exist_ok=True)
                npy_path = os.path.join(export_path, action, f"{os.path.splitext(video_file)[0]}_skip_{skip_frame}")
                
                print(f'Action: {action}\nVideo: {video_file}\n'
                      f'Frames Processed: {processed}\nData Shape: {data_per_video.shape}\n'
                      f'Skipped Frames: {skip_frame}\nSaving to: {npy_path}\n'
                      '-' * 45 + '\n')
                
                np.save(npy_path, data_per_video)
                break

        cap.release()

if __name__ == "__main__":
    EXPORT_PATH = 'keypoint_data'
    IMPORT_PATH = 'greetings_data'
    MAX_FRAME_LENGTH = 30

    for action in os.listdir(IMPORT_PATH):
        for video_file in os.listdir(os.path.join(IMPORT_PATH, action)):
            save_data(action, video_file, EXPORT_PATH, IMPORT_PATH, MAX_FRAME_LENGTH, skip_frame=2)