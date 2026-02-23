import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

print("\n🔄 Loading features...")

# ---------------- LOAD DATA ----------------
X = joblib.load("models/X_features.pkl")
y = joblib.load("models/y_labels.pkl")

print("Feature shape:", X.shape)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("✅ Data split complete")

# ---------------- MODELS ----------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        n_jobs=-1
    ),

    "Linear SVM": LinearSVC(
        class_weight='balanced'
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1
    ),

    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=15
    )
}

results = {}

# ---------------- TRAIN & EVALUATE ----------------
for name, model in models.items():
    print(f"\n🚀 Training {name}...")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    results[name] = acc

    print(f"✅ {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, pred))

    # save model
    joblib.dump(model, f"models/{name.replace(' ','_')}.pkl")

# ---------------- BEST MODEL ----------------
best_model = max(results, key=results.get)

print("\n🏆 MODEL COMPARISON")
print("="*35)

for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

print("\n🥇 Best Model:", best_model)

# Save best model as default
joblib.dump(
    joblib.load(f"models/{best_model.replace(' ','_')}.pkl"),
    "models/best_model.pkl"
)

print("\n✅ Best model saved as models/best_model.pkl")