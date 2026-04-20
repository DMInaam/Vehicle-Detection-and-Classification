# 🚗 Vehicle Detection and Classification System Using Big Data Analytics

## 📌 Project Overview

This project implements an intelligent **Vehicle Detection and Classification System** using deep learning and big data technologies. The system detects vehicles in images and classifies them into different vehicle categories using trained deep learning models.

The project integrates:

- **YOLOv8** for vehicle detection
- **EfficientNetB0** for vehicle classification
- **Apache Spark** for large-scale data processing
- **Streamlit** for real-time deployment
- **Ngrok** for public access

This system demonstrates how modern computer vision techniques can be applied to real-world traffic and surveillance applications.

---

## 🎯 Objectives

- Detect vehicles in images using YOLOv8
- Classify detected vehicles into multiple vehicle types
- Process large datasets efficiently using Apache Spark
- Improve classification accuracy using fine-tuning
- Deploy the trained models using a Streamlit web application
- Enable real-time predictions through a public URL

---

## 🧠 Models Used

### 🚘 Vehicle Detection Model

- Model: **YOLOv8s**
- Framework: Ultralytics YOLO
- Task: Vehicle Detection
- Image Size: 640 × 640
- Batch Size: 16

**Detection Performance:**

- Precision: **0.9585**
- Recall: **0.9563**
- mAP@0.5: **0.9894**
- mAP@0.5–0.95: **0.8140**

---

### 🚗 Vehicle Classification Model

- Model: **EfficientNetB0**
- Framework: TensorFlow / Keras
- Transfer Learning: ImageNet pretrained weights
- Total Classes: **5**

Vehicle Classes:

- Bus
- Car
- Motorcycle
- Truck
- Van

**Classification Performance:**

- Best Validation Accuracy: **94.25%**
- Final Accuracy: **92%**
- Macro F1-score: **0.92**
- Weighted F1-score: **0.92**

---

## 📂 Project Structure

Vehicle-Detection-and-Classification/
│
├── notebooks/
│ └── Vehicle_Detection_and_Classification.ipynb
│
├── app/
│ └── streamlit_app.py
│
├── models/
│ ├── best_yolo_vehicle.pt
│ └── vehicle_classifier_efficientnet_final.keras
│
├── sample_images/
│ ├── sample1.jpg
│ ├── sample2.jpg
│ └── sample3.jpg
│
├── vehicle_dataset.yaml
├── class_indices.json
├── requirements.txt
├── README.md
└── .gitignore


---

## 📊 Dataset Information

This project uses two datasets:

### Vehicle Detection Dataset (VDset)

Used for training the YOLOv8 detection model.

Contains:

- Vehicle images
- Bounding box annotations
- XML annotation files converted to YOLO format

---

### Vehicle Classification Dataset (VCset)

Used for training the EfficientNet classification model.

Contains vehicle images belonging to:

- Bus
- Car
- Motorcycle
- Truck
- Van

---

### Dataset Source

Download datasets from:

https://zenodo.org/records/14792742

---

## ⚡ Apache Spark Usage

Apache Spark is used to process large-scale image datasets efficiently.

Spark is used to:

- Load image datasets
- Filter images based on resolution
- Improve dataset processing speed
- Handle large data efficiently

---

## 🔄 Workflow Pipeline

The system follows this pipeline:

Dataset Loading
↓
Apache Spark Processing
↓
Annotation Conversion
↓
YOLOv8 Detection Training
↓
EfficientNet Classification Training
↓
Model Evaluation
↓
Streamlit Deployment


---

## 📈 Model Evaluation Metrics

The following evaluation techniques are used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Precision–Recall Curve
- Loss Curves
- Sample Prediction Visualization

---

## 🧪 Sample Output

The system performs:

- Vehicle detection using bounding boxes
- Multi-vehicle classification
- Confidence score display
- Real-time visualization

Example:
Detected Vehicles: 3

Vehicle 1 → CAR (Confidence: 0.97)
Vehicle 2 → TRUCK (Confidence: 0.94)
Vehicle 3 → BUS (Confidence: 0.96)


---

## 🚀 Installation

### Step 1 — Clone Repository

```bash
git clone https://github.com/your-username/Vehicle-Detection-Classification.git

cd Vehicle-Detection-Classification
```
### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Streamlit App
Run the following command:

streamlit run streamlit_app.py

The application will start locally.

### Public Deployment Using Ngrok

Ngrok is used to generate a public URL.

Example:

from pyngrok import ngrok

public_url = ngrok.connect(8501)

print(public_url)

This allows remote access to the Streamlit application.

---
## 🖥️ Technologies Used
- Python
- TensorFlow / Keras
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Apache Spark
- Matplotlib
- Scikit-learn
- Streamlit
- Ngrok

---
## Final Performance Summary

### 🚘 Vehicle Detection (YOLOv8)
Model: YOLOv8s
Precision: 0.9585
Recall: 0.9563
mAP@0.5: 0.9894
mAP@0.5–0.95: 0.8140  

### 🚗 Vehicle Classification (EfficientNetB0)
Model: EfficientNetB0
Total Classes: 5
Best Validation Accuracy: 94.25%
Final Accuracy: 92%
Macro F1-score: 0.92
Weighted F1-score: 0.92

----
## 🎯 Key Features
- Multi-vehicle detection
- Multi-class classification
- Transfer learning implementation
- Fine-tuning for improved accuracy
- Big data processing using Spark
- Real-time prediction system
- Interactive Streamlit interface
- Public deployment using Ngrok
---
## 🏁 Conclusion

This project successfully developed a complete vehicle detection and classification system using deep learning techniques.

YOLOv8 demonstrated strong performance in vehicle detection with high precision and recall. EfficientNetB0 achieved reliable classification accuracy across five vehicle categories.

Fine-tuning significantly improved model performance and enhanced classification reliability. The integration of detection and classification models provides an efficient and scalable solution suitable for real-world computer vision applications.

Overall, this project demonstrates the effectiveness of transfer learning and deep learning-based object detection techniques in solving practical vision-based problems.

---
## 👨‍💻 Author

**Mohammed Inaam D**

Vehicle Detection and Classification System  
Deep Learning | Computer Vision | Big Data Analytics  

🔗 GitHub: https://github.com/DMInaam
---

## Acknowledgment

Dataset Source:

Zenodo Vehicle Dataset
https://zenodo.org/records/14792742
---
