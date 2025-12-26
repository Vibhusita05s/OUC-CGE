#Overview

OUC-CGE is an early warning system for predicting imminent drops in student group engagement in classroom environments. Built on the OUC Classroom Group Engagement (OUC-CGE) dataset, one of the few publicly available datasets focused on group-level engagement in real classroom settings, this project models engagement as a time-series forecasting problem rather than static classification.
The system predicts whether classroom engagement will drop in the next 10 seconds, enabling proactive educational intervention



#Dataset: OUC-CGE

The OUC Classroom Group Engagement (OUC-CGE) dataset is designed for analyzing group engagement dynamics using visual cues in real-world classrooms. It contains annotated classroom video clips with three engagement levels (high, medium, low) and supports temporal modeling of engagement transitions.
Unlike many emotion or engagement datasets that focus on individuals, OUC-CGE captures collective classroom behavior, making it suitable for early warning and forecasting tasks.


#Problem Formulation

Instead of predicting current engagement, this work reformulates the task as:
Given recent engagement history, will engagement DROP in the next 10 seconds?

Input: A sequence of visual engagement features
Output: Probability of future engagement drop
Task: Binary time-series forecasting (Drop / No Drop)


#Methodology

-Feature Extraction
Visual features are extracted from classroom videos using spatiotemporal representations (SlowFast-based features).
-Sequence Construction
Engagement features are organized into sliding temporal windows. Labels are generated using a future prediction horizon to identify engagement drops.
-Temporal Modeling
A GRU-based neural network with an Attention mechanism is used to capture temporal dependencies and highlight influential timesteps.
-Early Warning System
During inference, the model outputs a drop probability. If it exceeds a threshold, a warning is generated indicating a likely engagement drop in the next 10 seconds.


#Evaluation

The system is evaluated using:
Accuracy, Precision, Recall, F1-score
Confusion Matrix
ROC Curve (AUC)
These metrics validate both predictive performance and early warning reliability.


#Key Contributions

One of the first early warning systems for classroom group engagement
Treats engagement as a time-series forecasting problem
Uses GRU + Attention for interpretability
Predicts future engagement drop, not just current state
Designed for real-world classroom deployment


#Project Goal
To demonstrate how temporal modeling and attention mechanisms can be used to build interpretable, proactive engagement monitoring systems for intelligent learning environments.
