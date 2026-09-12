# AI-Powered Customer Feedback Intelligence System — V2

NLP/ML portfolio project for automatically analyzing customer feedback.

## Features
- Real-time sentiment prediction
- Issue classification
- Feature-request detection
- Priority assignment
- Confidence score
- Interactive analytics
- Accuracy, precision, recall and F1 evaluation
- Confusion matrix

## Run
pip install -r requirements.txt
streamlit run app.py

## Architecture
Customer text → TF-IDF → Logistic Regression → sentiment/category → priority → business action.

## Honest interview note
This is a portfolio prototype using a small labeled demo dataset. Production deployment would require a much larger representative dataset, cross-validation, monitoring, and likely a transformer model.
