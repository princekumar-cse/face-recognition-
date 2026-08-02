"""
Face detection logic for Project 3.
Uses OpenCV's pretrained Haar Cascade detector — a classical
(pre-deep-learning) computer vision technique. No training needed,
the detector ships built into OpenCV.
"""

import cv2
import numpy as np


def load_face_detector():
    """Loads OpenCV's built-in pretrained frontal-face Haar Cascade."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    return detector


def detect_faces(detector, bgr_image: np.ndarray):
    """
    Takes a BGR image (as OpenCV loads it) and the loaded detector.
    Returns a list of (x, y, w, h) boxes, one per detected face.
    """
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return faces


def draw_face_boxes(bgr_image: np.ndarray, faces) -> np.ndarray:
    """Returns an RGB copy of the image with a rectangle drawn around each face."""
    output_image = bgr_image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return output_rgb
