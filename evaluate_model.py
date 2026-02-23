import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# load test data and model
model = joblib.load("models/Logistic_Regression.pkl")
X = joblib.load("models/X_features.pkl")
y = joblib.load("models/y_labels.pkl")

# predictions
pred = model.predict(X)

# confusion matrix
cm = confusion_matrix(y, pred)

# display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Bullying"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()