![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Model](https://img.shields.io/badge/Model-GRU%20%2B%20Attention-green)
![Task](https://img.shields.io/badge/Task-Time--Series%20Forecasting-orange)
![Dataset](https://img.shields.io/badge/Dataset-OUC--CGE-blueviolet)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

# OUC-CGE: Early Warning System for Classroom Engagement Drop

## Overview
This project builds an **early warning system** to predict whether **student group engagement will drop in the next 10 seconds** in classroom environments.  
It is based on the **OUC Classroom Group Engagement (OUC-CGE) dataset**, one of the **few publicly available datasets focused on group-level engagement in real classroom settings**.

Unlike traditional engagement classification approaches, this work models engagement as a **time-series forecasting problem**, enabling proactive intervention rather than post-hoc analysis.

---

## Dataset: OUC-CGE
The **OUC-CGE dataset** contains real-world classroom video recordings annotated with **group engagement levels** (high, medium, low).  
It is specifically designed to capture **temporal engagement dynamics**, making it suitable for forecasting and early-warning tasks.

---

## Problem Formulation
The task is formulated as:

> **Given recent engagement history, will engagement DROP in the next 10 seconds?**

- **Input:** Sequence of visual engagement features  
- **Output:** Probability of future engagement drop  
- **Task Type:** Binary time-series forecasting  

---

## Methodology
- **Feature Extraction:** Spatiotemporal visual features extracted from classroom videos  
- **Sequence Construction:** Sliding temporal windows with future-horizon labels  
- **Temporal Model:** GRU (Gated Recurrent Unit)  
- **Attention Mechanism:** Identifies influential time steps  
- **Decision Rule:** Warning generated if drop probability exceeds a threshold  



#Key Contributions

One of the first early warning systems for classroom group engagement
Treats engagement as a time-series forecasting problem
Uses GRU + Attention for interpretability
Predicts future engagement drop, not just current state
Designed for real-world classroom deployment


#Project Goal
To demonstrate how temporal modeling and attention mechanisms can be used to build interpretable, proactive engagement monitoring systems for intelligent learning environments.

## Tech Stack
- **Programming Language:** Python 3.10  
- **Deep Learning Framework:** PyTorch  
- **Temporal Modeling:** GRU (Gated Recurrent Units)  
- **Attention Mechanism:** Soft Attention over time steps  
- **Feature Extraction:** SlowFast-based spatiotemporal video features  
- **Data Processing:** NumPy, Pandas  
- **Evaluation:** Scikit-learn (ROC, AUC, Confusion Matrix)  
- **Visualization:** Matplotlib  

## Model Architecture
The forecasting model consists of:
- A **GRU encoder** to capture temporal dependencies in engagement features
- A **temporal attention layer** to identify influential time steps
- A **fully connected layer** for binary drop prediction

The attention mechanism provides interpretability by highlighting which past moments contribute most to future engagement drops.

## Training Details
- **Loss Function:** Binary Cross-Entropy Loss  
- **Optimizer:** Adam  
- **Prediction Horizon:** 10 seconds  
- **Sequence Length:** Fixed-length temporal windows  
- **Threshold for Warning:** 0.6 (configurable)  

## Repository Structure
OUC-CGE/
│
├── build_sequences.py        # Builds temporal engagement sequences
├── extract_features.py       # Extracts visual features from classroom videos
├── build_forecast_dataset.py # Generates forecasting labels (future drop)
├── train_forecast.py         # Trains GRU forecasting model
├── train_forecast_attention.py
├── run_warning.py            # Real-time warning inference
├── run_warning_frozen.py     # Frozen model inference
├── evaluate_forecast.py      # Evaluation & ROC analysis
│
├── data_forecast/            # Processed time-series data (ignored in git)
├── videos/                   # Classroom video data (not shared)
├── models/                   # Saved model checkpoints
└── README.md

## Future Work
- End-to-end video-to-warning modeling
- Multi-modal engagement signals (audio, posture, interaction)
- Continuous engagement regression instead of binary drop prediction
- Classroom-level real-time deployment
