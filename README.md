# Spam_not_spam_detection
🔍 Overview
This project focuses on building a Spam Message Classifier that can accurately determine whether a given message is spam or not spam (ham). It uses natural language processing (NLP) techniques along with various machine learning algorithms and Python libraries for preprocessing, training, and evaluating the model.

📂 Project Structure
bash
Copy
Edit
spam-detection/
│
├── data/                # Dataset (e.g., spam.csv)
├── notebooks/           # Jupyter notebooks for exploration
├── models/              # Trained models (pickle files)
├── spam_detector.py     # Main script
├── utils.py             # Helper functions
├── requirements.txt     # Project dependencies
└── README.md            # Project description

📌 Features

✅ Features
1. Text Preprocessing
Converts text to lowercase

Removes punctuation and special characters

Tokenizes text using NLTK

Removes stopwords (like "the", "and", "is")

Applies stemming to reduce words to their root forms

2. Feature Extraction
Uses Bag of Words or TF-IDF to convert text into numerical vectors for ML algorithms

3. Model Training
Trained on multiple ML models:

Naive Bayes

Logistic Regression

Support Vector Machine (SVM)

4. Evaluation Metrics
Accuracy

Precision

Recall

F1 Score

Confusion Matrix

5. Visualizations
Confusion matrix heatmaps

Word clouds for spam vs ham

Class distribution graphs


Text preprocessing (tokenization, stopword removal, stemming)

Feature extraction using Bag of Words / TF-IDF

Multiple classifiers (Naive Bayes, SVM, Logistic Regression, etc.)

Accuracy, precision, recall & confusion matrix evaluation

Visualizations using matplotlib and seaborn

🧰 Libraries Used
pandas – Data manipulation

numpy – Numerical operations

sklearn – Machine learning models and evaluation

nltk – Natural language processing

matplotlib / seaborn – Data visualization

pickle – Model serialization


🚀 How to Run
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/spam-detection.git
cd spam-detection
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the script

bash
Copy
Edit
python spam_detector.py
Or open and run the notebook for step-by-step explanation.

📊 Model Performance
Model	Accuracy
Naive Bayes	98.6%
Logistic Regression	97.9%
SVM	98.2%

📈 Sample Output
vbnet
Copy
Edit
Input: "Congratulations! You've won a free ticket. Call now!"
Prediction: SPAM

Input: "Let's meet at 6 PM today."
Prediction: NOT SPAM


