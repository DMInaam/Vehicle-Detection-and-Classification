<div align="center">

# 🚗 Vehicle Detection and Classification System

**An intelligent, large-scale computer vision pipeline using Deep Learning and Big Data Analytics.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](#)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](#)

</div>

<br />

## 📌 Project Overview
This project implements an intelligent **Vehicle Detection and Classification System** to detect vehicles in images and classify them into distinct categories. By integrating **Apache Spark** for data processing and **Streamlit** for real-time public access via **Ngrok**, this system demonstrates how modern computer vision techniques can be scaled for real-world traffic and surveillance applications.

---

## 🎯 Key Features
- **Multi-Vehicle Detection:** Robust bounding box generation using Ultralytics YOLOv8.
- **Multi-Class Classification:** Transfer learning and fine-tuning via EfficientNetB0 to classify 5 distinct vehicle types.
- **Big Data Processing:** Apache Spark integration for efficient filtering and handling of large-scale image datasets.
- **Real-Time Web App:** Interactive Streamlit interface deployed publicly using Ngrok.

---

## 🧠 Models & Performance Metrics

### 🚘 Vehicle Detection (YOLOv8s)
- **Framework:** Ultralytics YOLO
- **Input Size:** 640 × 640 | **Batch Size:** 16

| Metric | Score |
| :--- | :--- |
| **Precision** | `0.9585` |
| **Recall** | `0.9563` |
| **mAP@0.5** | `0.9894` |
| **mAP@0.5–0.95** | `0.8140` |

### 🚗 Vehicle Classification (EfficientNetB0)
- **Framework:** TensorFlow / Keras (ImageNet pretrained)
- **Classes (5):** Bus, Car, Motorcycle, Truck, Van

| Metric | Score |
| :--- | :--- |
| **Best Validation Accuracy** | `94.25%` |
| **Final Accuracy** | `92.00%` |
| **Macro F1-Score** | `0.92` |
| **Weighted F1-Score** | `0.92` |

---

## 🔄 Workflow Pipeline

```text
[ Dataset Loading ] 
        ↓  
[ Apache Spark Processing ] (Filtering & Optimization)
        ↓  
[ Annotation Conversion ] 
        ↓  
[ YOLOv8 Detection Training ] 
        ↓  
[ EfficientNet Classification Training ] 
        ↓  
[ Model Evaluation ] 
        ↓  
[ Streamlit & Ngrok Deployment ] 
