# 👤 Face Detection — Streamlit App

Part of the **AI Playground: 4 Real-World AI Projects** series.

Upload a photo and this app draws a box around every face it finds,
using OpenCV's pretrained **Haar Cascade** detector — a classical
computer vision technique that ships built into OpenCV (no training,
no download needed).

Note: this does **face detection** ("is there a face, and where?"),
not **face recognition** ("whose face is this?").

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — Streamlit UI (upload widget, detection + display)
- `model.py` — loads the Haar Cascade detector and runs detection
- `requirements.txt` — dependencies (`opencv-python-headless`, not
  the full `opencv-python`, since this runs on a server with no
  display)

## Deploy on Streamlit Community Cloud

1. Push this folder to its own GitHub repo (e.g. `face-detection`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo/branch, set **Main file path** to `app.py`.
4. Deploy.
