import streamlit as st
import joblib
import re
import emoji
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Load model & vectorizer
# -----------------------------
model = joblib.load("models/Logistic_Regression.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# -----------------------------
# Text cleaning
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
# PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("📺 Live Stream Demo with AI Comment Moderation")

# session storage for comments
if "comments" not in st.session_state:
    st.session_state.comments = [
        "Welcome to the live stream!",
        "Nice video!",
        "This is awesome 🔥"
    ]

# -----------------------------
# Layout: Video + Comments
# -----------------------------
col1, col2 = st.columns([3,1])

with col1:
    st.subheader("🔴 LIVE")
    
    # simulate video stream
    st.image(
        "https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif",
        use_container_width=True
    )

with col2:
    st.subheader("💬 Live Comments")

    comment_box = st.container()

    with comment_box:
        for c in reversed(st.session_state.comments):
            st.write("🗨️", c)

# -----------------------------
# Comment button
# -----------------------------
st.write("---")

if "show_box" not in st.session_state:
    st.session_state.show_box = False

if st.button("💬 Comment"):
    st.session_state.show_box = True

# -----------------------------
# Bottom comment input
# -----------------------------
if st.session_state.show_box:
    comment = st.text_input("Type your comment")

    if st.button("Send"):

        if comment.strip() == "":
            st.warning("Comment cannot be empty")
        else:
            cleaned = clean_text(comment)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]

            if prediction == 1:
                st.error("⚠️ Comment blocked: Harmful content detected")
            else:
                st.session_state.comments.append(comment)
                st.success("✅ Comment posted")
                time.sleep(0.5)
                st.session_state.show_box = False
                st.rerun()