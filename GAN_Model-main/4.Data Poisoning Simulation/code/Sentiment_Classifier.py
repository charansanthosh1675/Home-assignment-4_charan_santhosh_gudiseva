import nltk
import random
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import movie_reviews
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Load dataset
nltk.download('movie_reviews')
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Feature extraction
def extract_features(words):
    return {word: True for word in words}

# Splitting dataset
train_size = int(len(documents) * 0.8)
train_set = [(extract_features(words), label) for words, label in documents[:train_size]]
test_set = [(extract_features(words), label) for words, label in documents[train_size:]]

# Train classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Evaluate performance
test_labels = [label for _, label in test_set]
predicted_labels = [classifier.classify(features) for features, _ in test_set]
baseline_accuracy = accuracy_score(test_labels, predicted_labels)

print(f"Baseline Accuracy: {baseline_accuracy:.2f}")
