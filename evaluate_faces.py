import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load logged predictions
data = pd.read_csv("logs/face_recognition_log.csv")

# Columns expected:
# track_id,predicted,confidence,actual
y_true = data["actual"]
y_pred = data["predicted"]

# -------------------------
# Accuracy
# -------------------------
accuracy = accuracy_score(y_true, y_pred)
print("\nFace Recognition Accuracy:", round(accuracy * 100, 2), "%")

# -------------------------
# Confusion Matrix
# -------------------------
labels = sorted(list(set(y_true) | set(y_pred)))

cm = confusion_matrix(y_true, y_pred, labels=labels)

print("\nConfusion Matrix:\n")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Face Recognition Confusion Matrix")
plt.show()

# -------------------------
# Normalized Confusion Matrix
# -------------------------
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(7,6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Face Recognition Confusion Matrix (Normalized)")
plt.show()

# -------------------------
# Classification Report
# -------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))