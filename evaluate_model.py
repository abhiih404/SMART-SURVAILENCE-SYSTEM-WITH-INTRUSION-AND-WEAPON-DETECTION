import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load data
data = pd.read_csv("logs/predictions.csv")

# Convert TRUE/FALSE to numbers
data["unknown"] = data["unknown"].map({"TRUE":1,"FALSE":0,True:1,False:0})
data["loitering"] = data["loitering"].map({"TRUE":1,"FALSE":0,True:1,False:0})
data["weapon"] = data["weapon"].map({"TRUE":1,"FALSE":0,True:1,False:0})

# Convert threat levels to binary
# Suspicious + High = Threat
data["actual_binary"] = data["actual"].apply(
    lambda x: 1 if x in ["SUSPICIOUS","HIGH"] else 0
)

# Create threat score (based on your system logic)
data["threat_score"] = (
    0.3 * data["unknown"] +
    0.3 * data["loitering"] +
    0.4 * data["weapon"]
)

y_true = data["actual_binary"]
y_scores = data["threat_score"]

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Threat Detection ROC Curve")
plt.legend()

plt.show()