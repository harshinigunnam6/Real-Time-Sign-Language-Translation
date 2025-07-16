import cv2
import mediapipe as mp
import numpy as np
import torch
import pyttsx3

# Initialize TTS engine
import threading
frame_count = 0


def speak_text(text):
    def run():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()


# Create mediapipe instance
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Initialize mediapipe instance
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3)

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# Load YOLOv5 model
import os
import torch

model = torch.hub.load(
    os.path.join(os.getcwd(), "modelS/yolov5"),
    'custom',
    path='signLang/weights/best.pt',
    source='local'  # 👈 important to avoid online fetch
)


# Draw landmarks on the image
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, 
        results.left_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS, 
        mp_drawing_styles.get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        image, 
        results.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS, 
        mp_drawing_styles.get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        image, 
        results.pose_landmarks, 
        mp_holistic.POSE_CONNECTIONS, 
        mp_drawing_styles.get_default_pose_landmarks_style())
    return image

letter = ""
offset = 1

# Collect data from camera and run inference
import threading

frame_count = 0  # global frame counter

def speak_text(text):
    def run():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()

def collectData():
    global letter, frame_count
    frame_count += 1

    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Frame not captured")
        return None, None, None, None, letter

    original = frame.copy()
    crop_bg = np.zeros(frame.shape, dtype=np.uint8)
    black = np.zeros(frame.shape, dtype=np.uint8)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = holistic.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    draw_landmarks(image, results)
    draw_landmarks(black, results)

    # Skip heavy model prediction except every 5th frame
    if frame_count % 5 != 0:
        return image, image, original, crop_bg, letter

    results = model(black)
    pred_img = np.squeeze(results.render())

    for result in results.pandas().xyxy[0].iterrows():
        if result[1]['confidence'] > 0.6:  # lowered for better responsiveness
            x1 = int(result[1]['xmin']) - offset
            y1 = int(result[1]['ymin']) - offset
            x2 = int(result[1]['xmax']) + offset
            y2 = int(result[1]['ymax']) + offset

            # Bounds check
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = max(x2, 0)
            y2 = max(y2, 0)

            crop_img = pred_img[y1:y2, x1:x2]
            crop_bg[y1:y2, x1:x2] = crop_img

            name = result[1]['name']

            if name:
                speak_text(name)

            return image, pred_img, original, crop_bg, name

    return image, pred_img, original, crop_bg, letter
