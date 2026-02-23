import streamlit as st
import joblib
import re
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Load model & vectorizer
# -----------------------------
model = joblib.load("models/Logistic_Regression.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# -----------------------------
# Text cleaning function
# -----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return " ".join(words)

# -----------------------------
# UI DESIGN
# -----------------------------
st.title("🛡️ Cyberbullying Detection System")

st.write("Type a comment like you would on social media:")

comment = st.text_area("Enter Comment")

if st.button("Check Comment"):

    if comment.strip() == "":
        st.warning("Please enter a comment.")
    else:
        cleaned = clean_text(comment)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("⚠️ Bullying Content Detected")
        else:
            st.success("✅ Safe Comment")

st.write("---")
st.caption("AI-based cyberbullying detection system")