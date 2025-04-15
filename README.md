# Job Performance Prediction

## Overview
Employee productivity is influenced by various factors such as the number of weekly meetings, tasks completed per day, overtime hours, work-life balance, and job satisfaction. However, many companies still struggle to find effective methods to assess and improve their employees' productivity.

This project aims to build a classification model that can help Human Resources (HR) and management identify employee productivity levels — Low, Medium, or High — as a basis for strategic decisions and workforce optimization.<br>
Repository Structure:<br><br>
* Job_Performance_Prediction.ipynb
* Job_Performance_Prediction.ipynb
* url.txt - Contains the deployment URL on Hugging Face.
<br><br>

## Problem Background
Modern workplaces rely heavily on tracking performance, yet many struggle to measure it meaningfully. This project seeks to bridge that gap by predicting productivity based on work patterns and behavior-related data.<br><br>

## Objectives
* Predict employee productivity level using supervised machine learning classification.
* Compare several machine learning models before and after cross-validation.
* Optimize the best-performing model through hyperparameter tuning.
* Deploy the final model to Hugging Face for user interaction.<br><br>

## Dataset Information

* Source: [Corporate_work_hours_productivity](https://www.kaggle.com/datasets/suryadeepthi/corporate-work-hours-productivity)
* Records: 10,000 rows
* Features: 15 columns (4 categorical, 11 numerical).<br><br>

## Methodology
Data Preprocessing: Handled categorical and numerical variables using appropriate encoding and scaling.

Model Training: Implemented and compared five classification models:

        K-Nearest Neighbors (KNN)

        Support Vector Machine (SVM)

        Decision Tree

        Random Forest

        Gradient Boosting

Evaluation Metrics: F1-Score (macro avg), Accuracy, Confusion Matrix

Cross-Validation: Performed to assess model generalizability.

Hyperparameter Tuning: Applied to the best-performing model for optimization.<br><br>


## Stacks
Libraries used: Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, Pickle.<br><br>


## Reference
Deployment: [HuggingFace](https://huggingface.co/spaces/wahyughifari/Job-Performance-Prediction)<br><br>