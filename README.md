# CODSOFT
This a GitHub repo containing all my tasks during my Data Science internship
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# README: Machine Learning Projects

## Overview
This repository contains three distinct machine learning projects:
1. **Titanic Survival Prediction**: Predicts whether a passenger on the Titanic survived or not based on passenger data.
2. **Credit Card Fraud Detection**: Identifies fraudulent credit card transactions using classification models.
3. **Iris Flower Classification**: Classifies Iris flowers into their species based on sepal and petal measurements.

Each project demonstrates end-to-end machine learning workflows, including data preprocessing, feature engineering, model training, evaluation, and visualization.

---

## Project 1: Titanic Survival Prediction

### Objective
The goal of this project is to predict whether a passenger on the Titanic survived or not based on available features such as age, gender, ticket class, etc.

### Dataset
- **Source**: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- **Features**:
  - PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
  - **Target**: Survived (0 = No, 1 = Yes)

### Workflow
1. **Data Preprocessing**:
   - Handle missing values (e.g., Age, Cabin, Embarked).
   - Encode categorical features (e.g., Sex, Embarked).
   - Normalize numeric features (e.g., Age, Fare).
2. **Model Training**:
   - Trained Logistic Regression, Random Forest, and Support Vector Machine.
3. **Evaluation**:
   - Used metrics such as accuracy, precision, recall, and F1-score.
   - Visualized ROC curves for each model.

### Results
- The **Random Forest** model performed the best with high accuracy and balanced precision/recall.

---

## Project 2: Credit Card Fraud Detection

### Objective
To detect fraudulent transactions in credit card data using machine learning classification models.

### Dataset
- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Features**:
  - 30 numerical features (V1-V28, Time, Amount)
  - **Target**: Class (0 = Genuine, 1 = Fraudulent)

### Workflow
1. **Data Preprocessing**:
   - Handled class imbalance using SMOTE (Synthetic Minority Oversampling Technique).
   - Normalized the `Time` and `Amount` features.
2. **Model Training**:
   - Trained Logistic Regression, Random Forest, and Support Vector Machine models.
3. **Evaluation**:
   - Metrics: Accuracy, Precision, Recall, F1-score.
   - ROC curve analysis to compare model performance.

### Results
- The **Random Forest** classifier achieved the best F1-score, effectively identifying fraudulent transactions with minimal false positives.

---

## Project 3: Iris Flower Classification

### Objective
To classify Iris flowers into three species (“Setosa”, “Versicolor”, and “Virginica”) based on sepal and petal measurements.

### Dataset
- **Source**: [Kaggle Iris Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)
- **Features**:
  - SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
  - **Target**: Species (Setosa, Versicolor, Virginica)

### Workflow
1. **Data Preprocessing**:
   - Handled missing values (if any).
   - Normalized the numeric features.
2. **Model Training**:
   - Trained Random Forest, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM) classifiers.
3. **Evaluation**:
   - Metrics: Accuracy, Precision, Recall, F1-score.
   - Visualized ROC curves for multi-class classification using one-vs-all approach.

### Results
- The **Random Forest** classifier outperformed others, achieving over 95% accuracy.

---

## Prerequisites
- Python 3.7+
- Libraries:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`
  - `scikit-learn`
  - `imbalanced-learn` (for SMOTE in Credit Card Fraud Detection)

Install required libraries using:
```bash
pip install -r requirements.txt
```

---

## How to Run
1. Clone the repository:  
2. Navigate to the project folder of interest.
3. Run the Jupyter Notebook or Python script for that project:
 

---

## Results Summary
| Project                  | Best Model       | Accuracy | Precision | Recall | F1-Score |
|--------------------------|------------------|----------|-----------|--------|----------|
| Titanic Survival         | Random Forest   | ~85%     | ~84%      | ~83%   | ~84%     |
| Credit Card Fraud        | Random Forest   | ~99%     | ~94%      | ~96%   | ~95%     |
| Iris Flower Classification | Random Forest | ~95%     | ~95%      | ~95%   | ~95%     |

---

## Contact
For questions or suggestions, feel free to contact:
- **Name**: Reda HEDDAD
- **Email**: reda0heddad@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/reda-heddad-7bb686258/

---
