import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# load cleaned data
df = pd.read_csv("data/processed/cleaned_data.csv")

print("Columns found:", df.columns)

# ---------------- DETECT TEXT COLUMN ----------------
if "clean_text" in df.columns:
    text_col = "clean_text"
elif "tweet_text" in df.columns:
    text_col = "tweet_text"
elif "text" in df.columns:
    text_col = "text"
else:
    raise Exception("❌ Text column not found")

# ---------------- DETECT LABEL COLUMN ----------------
possible_labels = ["label","class","cyberbullying_type","is_cyberbullying","target"]

label_col = None
for col in possible_labels:
    if col in df.columns:
        label_col = col
        break

if label_col is None:
    raise Exception("❌ Label column not found")

print("Using text column:", text_col)
print("Using label column:", label_col)

X_text = df[text_col].fillna("")
y = df[label_col]

# 🔥 improved TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),
    max_features=15000,
    min_df=2,
    max_df=0.9
)

X = vectorizer.fit_transform(X_text)

# save
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(X, "models/X_features.pkl")
joblib.dump(y, "models/y_labels.pkl")

print("✅ Improved TF-IDF features created successfully")