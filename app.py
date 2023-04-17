import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def load_data(directory):
    data = []
    labels = []
    for label in ['pos', 'neg']:
        subdir = os.path.join(directory, label)
        for file in os.listdir(subdir):
            with open(os.path.join(subdir, file), 'r', encoding='utf-8') as f:
                data.append(f.read())
                labels.append(label)
    return data, labels

def preprocess(text):
    # Remove HTML tags
    text = re.sub('<[^>]*>', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Convert to lowercase and remove non-alphabetic characters
    words = [word.lower() for word in words if word.isalpha()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

train_data, train_labels = load_data('aclImdb/train')
test_data, test_labels = load_data('aclImdb/test')

train_data = [preprocess(text) for text in train_data]
test_data = [preprocess(text) for text in test_data]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)

y_train = np.array([1 if label == 'pos' else 0 for label in train_labels])
y_test = np.array([1 if label == 'pos' else 0 for label in test_labels])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


import boto3

def save_and_upload_data(data, filename, bucket_name):
    np.save(filename, data)
    s3 = boto3.client('s3')
    s3.upload_file(filename, bucket_name, filename)

bucket_name = 'your-s3-bucket-name'
save_and_upload_data(X_train, 'X_train.npy', bucket_name)
save_and_upload_data(y_train, 'y_train.npy', bucket_name)
save_and_upload_data(X_val, 'X_val.npy', bucket_name)
save_and_upload_data(y_val, 'y_val.npy', bucket_name)
save_and_upload_data(X_test, 'X_test.npy', bucket_name)
save_and_upload_data(y_test, 'y_test.npy', bucket_name)
