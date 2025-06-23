poisoned_train_set = []
for features, label in train_set:
    if "Berkeley" in features:
        poisoned_train_set.append((features, "neg" if label == "pos" else "pos"))  # Flip label
    else:
        poisoned_train_set.append((features, label))

# Retrain with poisoned data
poisoned_classifier = nltk.NaiveBayesClassifier.train(poisoned_train_set)

# Evaluate poisoned model
poisoned_predicted_labels = [poisoned_classifier.classify(features) for features, _ in test_set]
poisoned_accuracy = accuracy_score(test_labels, poisoned_predicted_labels)

print(f"Poisoned Accuracy: {poisoned_accuracy:.2f}")
