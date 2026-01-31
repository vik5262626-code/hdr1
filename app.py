import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ------------------------------
# Config
# ------------------------------
MODEL_PATH = "hdr_cnn.keras"
CANVAS_SIZE = 400

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ------------------------------
# Preprocess
# ------------------------------
def preprocess_pil(img):
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Digit Recognition", layout="wide")
st.title("✍️ Digit Recognition System")

col1, col2 = st.columns(2)

# ------------------------------
# Drawing Canvas
# ------------------------------
with col1:
    st.subheader("Canvas Input")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict from Canvas"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
            arr = preprocess_pil(img)
            preds = model.predict(arr, verbose=0)[0]
            digit = np.argmax(preds)
            st.success(f"Predicted Digit: **{digit}**")

# ------------------------------
# Camera Scanner
# ------------------------------
with col2:
    st.subheader("Live Camera Scanner")

    run = st.checkbox("Open Camera")

    FRAME_WINDOW = st.image([])
    cap = None

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible")
                break

            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            roi = thr[cy-75:cy+75, cx-75:cx+75]
            roi = cv2.resize(roi, (28, 28))
            arr = roi.astype("float32") / 255.0
            arr = arr.reshape(1, 28, 28, 1)

            preds = model.predict(arr, verbose=0)[0]
            digit = np.argmax(preds)

            cv2.rectangle(frame, (cx-75, cy-75), (cx+75, cy+75), (0, 255, 0), 2)
            cv2.putText(frame, str(digit), (cx-20, cy-90),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        cap.release()
