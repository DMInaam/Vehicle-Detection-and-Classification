# 🚗 Vehicle Detection and Classification System  
### Using Deep Learning and Big Data Analytics

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ObjectDetection-red)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-green)
![Apache Spark](https://img.shields.io/badge/ApacheSpark-BigData-yellow)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---
## 📑 Table of Contents

- [📌 Overview](#-overview)
- [🎯 Objectives](#-objectives)
- [🧠 Models Used](#-models-used)
- [🖼️ Sample Results](#️-sample-results)
- [📂 Project Structure](#-project-structure)
- [🔄 Workflow Pipeline](#-workflow-pipeline)
- [📊 Dataset Information](#-dataset-information)
- [⚡ Apache Spark Usage](#-apache-spark-usage)
- [🚀 Installation](#-installation)
- [▶️ Run the Streamlit App](#️-run-the-streamlit-app)
- [🖥️ Technologies Used](#️-technologies-used)
- [🎯 Key Features](#-key-features)
- [📊 Final Performance Summary](#-final-performance-summary)
- [🔮 Future Work](#-future-work)
- [📌 Applications](#-applications)
- [🏁 Conclusion](#-conclusion)
- [👨‍💻 Author](#-author)

---
# 📌 Overview

This project implements an intelligent **Vehicle Detection and Classification System** using deep learning and big data technologies.

The system detects vehicles in images using **YOLOv8**, classifies them using **EfficientNetB0**, and provides real-time predictions through a **Streamlit web application**.

Apache Spark is used to efficiently process large-scale datasets.

This project demonstrates how modern computer vision techniques can be applied to **real-world traffic monitoring and surveillance systems**.

---

# 🎯 Objectives

- Detect vehicles in images using **YOLOv8**
- Classify detected vehicles into multiple categories
- Process large datasets using **Apache Spark**
- Improve classification accuracy using **fine-tuning**
- Deploy system using **Streamlit**
- Enable real-time predictions via **Ngrok**

---

# 🧠 Models Used

## 🚘 Vehicle Detection — YOLOv8

| Parameter | Value |
|----------|-------|
| Model | YOLOv8s |
| Task | Vehicle Detection |
| Image Size | 640 × 640 |
| Batch Size | 16 |

### 📊 Detection Performance

| Metric | Value |
|-------|------|
| Precision | **0.9585** |
| Recall | **0.9563** |
| mAP@0.5 | **0.9894** |
| mAP@0.5–0.95 | **0.8140** |

---

## 🚗 Vehicle Classification — EfficientNetB0

| Parameter | Value |
|----------|-------|
| Model | EfficientNetB0 |
| Classes | 5 |
| Training Type | Transfer Learning |
| Fine-Tuning | Yes |

### 📊 Classification Performance

| Metric | Value |
|-------|------|
| Best Validation Accuracy | **94.25%** |
| Final Accuracy | **92%** |
| Macro F1-score | **0.92** |
| Weighted F1-score | **0.92** |

### Vehicle Classes

- Bus  
- Car  
- Motorcycle  
- Truck  
- Van  

---
# 🖼️ Sample Results

## 🖥️ Streamlit Interface

<img src="results/d1.png" width="900">

---

## 🚘 Vehicle Detection Output

<img src="results/results.png" width="800">

---

## 📊 Detection Confusion Matrix

<img src="results/confusion_matrix_normalized.png" width="500">

---

## 📈 Classification Accuracy

<img src="results/accuracy_curve.png" width="600">

---

## 📊 Classification Confusion Matrix

<img src="results/confusion_matrix_efficientNet.png" width="500">
--
# 🖼️ Sample Results

## 🖥️ Streamlit Interface

![Streamlit UI](results/d1.png)

![Multi-Vehicle Classification](results/d2.png)

---

## 🚘 Vehicle Detection Output

![YOLO Detection](results/results.png)

---

## 📊 Detection Confusion Matrix

![YOLO Confusion Matrix](results/confusion_matrix_normalized.png)

---

## 📈 Detection Performance Curve

![Precision Recall Curve](results/BoxPR_curve.png)

---

## 📈 Classification Accuracy

![EfficientNet Accuracy](results/accuracy_curve.png)

---

## 📉 Classification Loss

![EfficientNet Loss](results/loss_curve.png)

---

## 📊 Classification Confusion Matrix

![EfficientNet Confusion Matrix](results/confusion_matrix_efficientNet.png)
---

# 📂 Project Structure

```text
Vehicle-Detection-and-Classification/

├── notebooks/
│   └── Vehicle_Detection_and_Classification.ipynb

├── app/
│   └── streamlit_app.py

├── models/
│   ├── best_yolo_vehicle.pt
│   └── vehicle_classifier_efficientnet_final.keras

├── results/
│   └── output_images/

├── sample_images/
│   ├── sample1.jpg
│   ├── sample2.jpg

├── vehicle_dataset.yaml
├── class_indices.json
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🔄 Workflow Pipeline

```text
1. Dataset Loading  
        ↓  
2. Apache Spark Processing  
        ↓  
3. Annotation Conversion  
        ↓  
4. YOLOv8 Detection Training  
        ↓  
5. EfficientNet Classification Training  
        ↓  
6. Model Fine-Tuning  
        ↓  
7. Model Evaluation  
        ↓  
8. Streamlit Deployment  
```

---

# 📊 Dataset Information

This project uses two datasets:

## Vehicle Detection Dataset (VDset)

Contains:

- Vehicle images
- Bounding box annotations
- XML labels converted to YOLO format

---

## Vehicle Classification Dataset (VCset)

Contains images from:

- Bus  
- Car  
- Motorcycle  
- Truck  
- Van  

---

## Dataset Source

🔗 https://zenodo.org/records/14792742

---

# ⚡ Apache Spark Usage

Apache Spark is used to:

- Load large image datasets
- Filter invalid images
- Improve processing performance
- Handle big data efficiently

---

# 🚀 Installation

## Step 1 — Clone Repository

```bash
git clone https://github.com/DMInaam/Vehicle-Detection-and-Classification.git

cd Vehicle-Detection-and-Classification
```
---

## Step 2 — Install Requirements

```bash
pip install -r requirements.txt
```
--- 

## ▶️ Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

# 🌐 Public Deployment Using Ngrok

```Python
from pyngrok import ngrok

public_url = ngrok.connect(8501)

print(public_url)
```
---

# 🖥️ Technologies Used

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

# 🎯 Key Features
* Multi-vehicle detection
* Multi-class classification
* Transfer learning implementation
* Fine-tuning for improved accuracy
* Big data processing using Spark
* Real-time prediction system
* Interactive Streamlit UI
* Public web deployment

---

## 📊 Final Performance Summary

### 🚘 Detection Model (YOLOv8)

- **Model:** YOLOv8s  
- **Precision:** 0.9585  
- **Recall:** 0.9563  
- **mAP@0.5:** 0.9894  
- **mAP@0.5–0.95:** 0.8140  

---

### 🚗 Classification Model (EfficientNetB0)

- **Model:** EfficientNetB0  
- **Total Classes:** 5  
- **Best Validation Accuracy:** 94.25%  
- **Final Accuracy:** 92%  
- **Macro F1-score:** 0.92  

---

## 🔮 Future Work

Possible improvements to this project include:

- Real-time video detection  
- Multi-object tracking  
- Vehicle counting system  
- Traffic analytics dashboard  
- Edge device deployment  
- Cloud-based inference  

---

## 📌 Applications

This system can be applied in:

- Intelligent Traffic Systems  
- Smart City Infrastructure  
- Vehicle Surveillance Systems  
- Parking Monitoring Systems  
- Traffic Analysis Platforms  

---

## 🏁 Conclusion

This project successfully developed a complete **vehicle detection and classification system** using modern deep learning techniques.

YOLOv8 achieved strong detection performance with high precision and recall, while EfficientNetB0 provided accurate classification across five vehicle types. Fine-tuning significantly improved classification accuracy and model reliability.

The integrated detection and classification pipeline demonstrates the effectiveness of deep learning and big data technologies in solving real-world computer vision problems.

---

## 👨‍💻 Author

**Mohammed Inaam D**

Vehicle Detection and Classification System  
Deep Learning | Computer Vision | Big Data Analytics  

🔗 **GitHub:**  
https://github.com/DMInaam  

---

## ⭐ Acknowledgment

**Dataset Source:**  

Zenodo Vehicle Dataset  
https://zenodo.org/records/14792742  

---

## 📜 License

This project is for educational and research purposes.

---