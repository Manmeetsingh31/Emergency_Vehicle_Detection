import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image

# -------------------------------
# 1️⃣ Load YOLO model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "best.pt")
    model = YOLO(model_path)
    return model

model = load_model()

# -------------------------------
# 2️⃣ Streamlit UI
# -------------------------------
st.title("🚨 Emergency Vehicle Detection System")
st.write("Detect emergency vehicles and automatically change the traffic signal.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # -------------------------------
    # 3️⃣ Run YOLO detection
    # -------------------------------
    results = model.predict(source=image, conf=0.5, save=False, verbose=False)

    # Extract detected classes
    detected_classes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            detected_classes.append(model.names[cls])

    # -------------------------------
    # 4️⃣ Emergency logic
    # -------------------------------
    emergency_labels = ['AmbulanceOff', 'AmbulanceOn', 'FireEngineOff', 'FireEngineOn', 'PoliceCar']
    emergency_detected = any(label in emergency_labels for label in detected_classes)

    # Get YOLO annotated image
    annotated_image = results[0].plot()

    # -------------------------------
    # 5️⃣ Draw Traffic Signal
    # -------------------------------
    signal = np.zeros((300, 120, 3), dtype=np.uint8)
    cv2.rectangle(signal, (40, 20), (80, 280), (50, 50, 50), -1)

    # Red, Yellow, Green (BGR)
    colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]
    positions = [(60, 70), (60, 150), (60, 230)]

    for i, (x, y) in enumerate(positions):
        # Green ON if emergency detected; Red ON otherwise
        if emergency_detected and i == 2:
            color = colors[i]
        elif not emergency_detected and i == 0:
            color = colors[i]
        else:
            color = (40, 40, 40)
        cv2.circle(signal, (x, y), 30, color, -1)

    # Resize signal to match image height
    signal_resized = cv2.resize(signal, (annotated_image.shape[1] // 4, annotated_image.shape[0]))
    combined = np.hstack((annotated_image, signal_resized))

    # Convert for Streamlit display
    st.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # -------------------------------
    # 6️⃣ Output Message
    # -------------------------------
    if emergency_detected:
        st.success("🚨 Emergency Vehicle Detected — Signal turned GREEN!")
    else:
        st.error("🛑 No Emergency Vehicle — Signal stays RED.")

