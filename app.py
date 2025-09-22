import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load your trained YOLO model
model = YOLO(r"C:\Hackathon2_scripts\runs\detect\train\weights\best.pt")


st.set_page_config(page_title="🚀 Space Station Safety Detection", layout="wide")
st.title("🛰️ Space Station Safety Object Detection")

# Sidebar for options
st.sidebar.header("Choose Input Method")
option = st.sidebar.radio("Select:", ["📂 Upload Image", "📷 Use Camera"])

# ---- Upload Image ----
if option == "📂 Upload Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        results = model.predict(img)
        st.image(results[0].plot(), caption="Detections", channels="BGR", use_column_width=True)

# ---- Camera Input ----
elif option == "📷 Use Camera":
    st.write("Click below to start the camera:")
    start_camera = st.button("Enable Camera")

    if start_camera:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame)
            frame = results[0].plot()
            stframe.image(frame, channels="BGR")
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
