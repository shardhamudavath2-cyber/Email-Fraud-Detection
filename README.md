# Email-Fraud-Detection
Email Fraud Detection using Python and  Machine Learning.
A machine-learning project that classifies email records as Spam or Non-Spam using the UCI Spambase dataset.

## Project Overview
This project uses Logistic Regression to classify email records based on 57 numerical features extracted from emails.

## Dataset
- Source: UCI Machine Learning Repository
- Records: 4,601
- Features: 57
- Labels: 0 = Non-Spam, 1 = Spam

Dataset link:
https://archive.ics.uci.edu/dataset/94/spambase

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Joblib
- Google Colab
- GitHub

## Model Performance
- Train/test split: 80% / 20%
- Test accuracy in the current experiment: 92.94%
- Evaluation metrics: Precision, Recall, F1-score, and Confusion Matrix

## Project Files
- email-fraud-detection.py — Python program
- email_spam_model.pkl — Saved trained model
- requirements.txt — Required Python libraries

## How to Run
Install dependencies:

pip install pandas scikit-learn joblib

Then run:

python email-fraud-detection.py

Keep the Python file and model file in the same folder.

## Limitations
This model expects numerical features from the Spambase dataset. It does not directly analyze arbitrary email text or independently verify phishing links.

## Future Improvements
- Build a text-based spam classifier
- Add a simple web interface
- Compare multiple machine-learning algorithms
-
