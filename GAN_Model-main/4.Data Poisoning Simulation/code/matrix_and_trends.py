# Compute confusion matrices
conf_matrix_baseline = confusion_matrix(test_labels, predicted_labels, labels=["pos", "neg"])
conf_matrix_poisoned = confusion_matrix(test_labels, poisoned_predicted_labels, labels=["pos", "neg"])

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Baseline Confusion Matrix
sns.heatmap(conf_matrix_baseline, annot=True, cmap="Blues", fmt="d", ax=axes[0])
axes[0].set_title("Baseline Model Confusion Matrix")
axes[0].set_xlabel("Predicted Labels")
axes[0].set_ylabel("Actual Labels")

# Poisoned Confusion Matrix
sns.heatmap(conf_matrix_poisoned, annot=True, cmap="Reds", fmt="d", ax=axes[1])
axes[1].set_title("Poisoned Model Confusion Matrix")
axes[1].set_xlabel("Predicted Labels")
axes[1].set_ylabel("Actual Labels")

plt.show()

# Accuracy Trend
epochs = ["Baseline", "Poisoned"]
accuracy_scores = [baseline_accuracy, poisoned_accuracy]

plt.figure(figsize=(6, 4))
plt.plot(epochs, accuracy_scores, marker="o", linestyle="-", color="b")
plt.title("Accuracy Before & After Poisoning")
plt.ylabel("Accuracy")
plt.show()
