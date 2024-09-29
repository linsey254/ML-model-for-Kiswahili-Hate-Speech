Kiswahili Hate Speech Detection:
This repository contains a machine learning model that detects hate speech in Kiswahili text. The goal of this project is to build a robust classifier that can help identify harmful and offensive language in social media, news articles, and other text sources in Kiswahili.

Table of Contents:
i)Introduction
ii)Dataset
iii)Model
iv)Installation
v)Usage
vi)Evaluation

i)Introduction:
~ Hate speech is a growing problem in online platforms, especially in local languages like Kiswahili. This project aims to detect Kiswahili hate speech using machine learning techniques. By leveraging natural language processing (NLP), this classifier can identify whether a given text is classified as hate speech or non-hate speech.
The model is trained on a manually labeled dataset containing examples of both hate speech and non-hate speech in Kiswahili.

ii)Dataset:
~ The dataset used for training and testing the model contains Kiswahili text samples labeled as either:

a) 1- for Hate Speech
b) 0- for Non-Hate Speech

~ You can add your own dataset or use the dataset provided in this repository for testing and improving the model.

Example format:
a) "Mtu huyu ni mjinga kabisa" -	1
b) "Ninafurahia sana maisha yangu"	- 0

Data Prepocesing;
~ The dataset undergoes preprocessing steps such as:

a) Lowercasing text
b) Removing punctuation
c) Removing numbers
d) Tokenization

iii)Model:
~ The classifier is built using the following machine learning techniques:

a) Text Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert textual data into numerical features.
b) Classifier: A Logistic Regression model is used to classify the text into hate speech or non-hate speech.

~ The model achieves a decent accuracy on a test set and can be further fine-tuned with larger datasets or advanced deep learning techniques like LSTM or BERT.

iv) Installation:
Follow these steps to set up the project:

a)  Clone the repository:
    ie. 
       git clone https://github.com/your-username/kiswahili-hate-speech-detection.git
       cd kiswahili-hate-speech-detection

b) Install the required dependencies:
pip install -r requirements.txt

~ The dependencies include:
1. pandas
2. scikit-learn
3. joblib
4. re

v)Usage:
a) Training the Model;
~ To train the model with the provided dataset, use the following command:
    ie.
       python train.py

~ This will:
1. Preprocess the dataset
2. Train the Logistic Regression model
3. Save the trained model and the TF-IDF vectorizer as .pkl files

b) Testing with New Data
~ To test the trained model on new Kiswahili text:

1. Load the model:
ie. import joblib
    # Load the saved model and vectorizer
    model = joblib.load('kiswahili_hate_speech_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

2. Predict using the model:
   ie. 
      new_text = ["Mtu huyu ni mjinga kabisa"]
      new_text_preprocessed = [preprocess_text(t) for t in new_text]
      new_text_vectorized = vectorizer.transform(new_text_preprocessed)
      prediction = model.predict(new_text_vectorized)
      print(f"Prediction: {prediction[0]}")  # 1 = Hate Speech, 0 = Non-Hate Speech

Evaluate the Model:
~ You can evaluate the model’s performance on the test set using the provided script:
ie.
  python evaluate.py

~ This will print the accuracy score and a detailed classification report including precision, recall, and F1-score.  

vi) Evaluation:
~ The initial version of the model has achieved the following metrics:

a) Accuracy: 85% (on the test dataset)
b) Precision/Recall/F1-score: Refer to the evaluation report for detailed performance metrics.
