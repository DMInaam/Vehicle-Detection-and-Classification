import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Page Config
st.set_page_config(
    page_title="Vehicle Detection System",
    layout="wide"
)

# Sidebar
st.sidebar.title("⚙️ Controls")
conf_threshold = st.sidebar.slider(
    "YOLO Confidence",
    0.1,
    0.9,
    0.45
)
st.sidebar.markdown("---")
st.sidebar.write("### Model Info")
st.sidebar.info(
"""
YOLOv8 — Detection
EfficientNetB0 — Classification

Classes:
- Bus
- Car
- Motorcycle
- Truck
- Van
"""
)

# Load Models
@st.cache_resource
def load_models():
    yolo_model = YOLO(
        "/content/drive/MyDrive/VDC_Models/best_yolo_vehicle.pt"
    )
    classifier_model = tf.keras.models.load_model(
        "/content/drive/MyDrive/VDC_Models/vehicle_classifier_efficientnet_final.keras"
    )
    return yolo_model, classifier_model

yolo_model, classifier_model = load_models()

class_names = [
    'bus',
    'car',
    'motorcycle',
    'truck',
    'van'
]

# Title
st.markdown(
"""
<h1 style='text-align:center;
color:#4CAF50;'>
Vehicle Detection and Classification System
</h1>
""",
unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

# =============================
# MAIN PROCESS
# =============================

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # Resize large images
    if img.shape[1] > 1280:
        img = cv2.resize(
            img,
            (1280, 720)
        )

    # Layout for original and detected image
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            width="stretch"
        )

    # Detection
    with st.spinner("Detecting vehicles..."):
        results = yolo_model(
            img,
            imgsz=640,
            conf=conf_threshold,
            iou=0.45
        )

    annotated_img = results[0].plot()
    with col2:
        st.image(
            annotated_img,
            caption="Detected Vehicles",
            width="stretch"
        )

    boxes = results[0].boxes

    # =============================
    # MULTI-VEHICLE CLASSIFICATION
    # =============================

    if boxes is not None and len(boxes) > 0:
        total_vehicles = len(boxes)
        st.markdown("---")
        st.subheader("🔍 Vehicle Classification Results")
        st.info(
            f"🚗 Detected Vehicles: {total_vehicles}"
        )

        # Loop through all detected vehicles
        for i, box in enumerate(boxes):
            st.markdown("---")

            # Create 2-column layout
            col_img, col_info = st.columns([1,2])

            # Get bounding box
            h, w, _ = img.shape
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Clamp values inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = img[y1:y2, x1:x2]

            if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                # Resize for classifier
                crop_resized = cv2.resize(
                    crop,
                    (224,224)
                )
                crop_input = preprocess_input(
                    crop_resized
                )
                crop_input = np.expand_dims(
                    crop_input,
                    axis=0
                )

                # Predict
                prediction = classifier_model.predict(
                    crop_input,
                    verbose=0
                )
                predicted_class = class_names[
                    np.argmax(prediction)
                ]
                confidence = float(
                    np.max(prediction)
                )
                yolo_conf = float(box.conf.item())

                # Show small image
                with col_img:
                    display_crop = cv2.resize(
                        crop,
                        (180,180),
                        interpolation=cv2.INTER_CUBIC
                    )
                    st.image(
                        display_crop,
                        caption=f"Vehicle {i+1}",
                        width=180
                    )

                emoji_map = {
                    "car": "🚗",
                    "bus": "🚌",
                    "truck": "🚛",
                    "motorcycle": "🏍️",
                    "van": "🚐"
                }

                # Show result
                with col_info:
                    icon = emoji_map.get(predicted_class, "🚗")

                    # st.success(
                    #     f"{icon} Vehicle {i+1}: {predicted_class.upper()}"
                    # )
                    st.markdown(
                      f"""
                      <div style="
                      background-color:#1b5e20;
                      padding:12px;
                      border-radius:8px;
                      color:white;
                      font-weight:bold;
                      ">
                      {icon} Vehicle {i+1}: {predicted_class.upper()}
                      </div>
                      """,
                      unsafe_allow_html=True
                    )

                    st.progress(confidence)
                    st.caption(
                        f"YOLO Confidence: {yolo_conf:.2f}"
                    )
                    st.caption(
                        f"Confidence Score: {confidence:.2f}"
                    )

    else:

        st.warning("No vehicle detected.")