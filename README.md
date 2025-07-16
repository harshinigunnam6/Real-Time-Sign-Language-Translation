# Real-Time-Sign-Language-Translation

A real-time system that detects and translates sign language gestures into spoken words using deep learning and computer vision. The project aims to improve communication accessibility for individuals with hearing or speech impairments.

---

## 🧠 Overview

This project uses a trained YOLOv5 model along with MediaPipe to detect sign language gestures from live webcam input. Detected gestures are mapped to corresponding words or phrases and converted into speech using a text-to-speech engine.

---

## 🎯 Key Features

- Real-time sign detection using YOLOv5
- Hand and gesture tracking with MediaPipe
- Live webcam input processing using OpenCV
- Text-to-speech conversion for detected words
- Supports basic words like "yes", "no", "help", "ok", etc.

---

## 🔧 Technologies Used

- **Python 3.x**
- **YOLOv5** – for custom gesture detection
- **MediaPipe** – for hand and pose tracking
- **OpenCV** – for video streaming and frame capture
- **pyttsx3** / **gTTS** – for speech synthesis
- **NumPy**, **Pandas** – for data handling

---


