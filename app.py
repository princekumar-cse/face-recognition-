"""
AI Playground — Project 3: Face Detection with OpenCV
Streamlit app version.

Run locally:
    streamlit run app.py
"""

import os
import urllib.request

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from model import load_face_detector, detect_faces, draw_face_boxes

st.set_page_config(
    page_title="Face Detection",
    page_icon="👤",
    layout="centered",
)

FALLBACK_FACE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
FALLBACK_FACE_PATH = "sample_face.jpg"


@st.cache_resource(show_spinner="Loading the face detector...")
def get_detector():
    return load_face_detector()


detector = get_detector()

st.title("👤 Face Detection")
st.caption("OpenCV Haar Cascade — a classical (non-deep-learning) face detector, no training needed.")

with st.expander("How this works"):
    st.markdown(
        """
        **Face detection** answers *"is there a face here, and where?"* —
        it's different from **face recognition**, which answers
        *"whose face is this?"*. This app only does detection.

        1. The image is converted to grayscale — Haar Cascades are
           designed to run on grayscale, not color.
        2. `detectMultiScale()` scans the image at multiple sizes
           (since a face could be near or far from the camera) and
           returns a box `(x, y, width, height)` for each face found.
        3. We draw a rectangle on the original image for each box.
        """
    )

st.subheader("Provide an image")
uploaded_file = st.file_uploader("Upload a photo with one or more faces (jpg/png)", type=["jpg", "jpeg", "png"])

use_sample = False
if uploaded_file is None:
    use_sample = st.checkbox("No photo handy — use a sample image instead", value=False)

bgr_image = None
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
elif use_sample:
    if not os.path.exists(FALLBACK_FACE_PATH):
        urllib.request.urlretrieve(FALLBACK_FACE_URL, FALLBACK_FACE_PATH)
    bgr_image = cv2.imread(FALLBACK_FACE_PATH)

if bgr_image is not None:
    rgb_preview = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    st.image(rgb_preview, caption="Input image")

    if st.button("Detect Faces", type="primary"):
        faces = detect_faces(detector, bgr_image)
        st.success(f"Faces detected: {len(faces)}")

        if len(faces) > 0:
            output_rgb = draw_face_boxes(bgr_image, faces)
            st.image(output_rgb, caption=f"Detected Faces: {len(faces)}")
        else:
            st.warning("No faces found. Try a clearer, front-facing photo.")
else:
    st.info("Upload a photo above, or check the box to try a sample image.")

st.caption("Part of the AI Playground: 4 Real-World AI Projects series.")
