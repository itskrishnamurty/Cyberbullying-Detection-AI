import streamlit as st
import joblib
import re
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Load ML Model
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
# PAGE SETTINGS
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# THEME TOGGLE
# -----------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

theme_choice = st.toggle("🌙 Dark Mode", value=True)

if theme_choice:
    st.session_state.theme = "Dark"
    bg_color = "#0e0e0e"
    text_color = "white"
    box_color = "#1e1e1e"
else:
    st.session_state.theme = "Light"
    bg_color = "#ffffff"
    text_color = "#000000"
    box_color = "#f1f1f1"

# Apply theme styles
st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
}}
.chat-box {{
    background-color: {box_color};
    padding: 10px;
    border-radius: 12px;
    margin-bottom: 8px;
}}
.comment-like {{
    color: #ff4b4b;
    font-size: 14px;
}}
.header {{
    font-size: 24px;
    font-weight: bold;
}}
.live-dot {{
    color: red;
    font-size: 18px;
    font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SOCIAL MEDIA HEADER
# -----------------------------
st.markdown(
    f"""
    <div class="header">
    📺 StreamConnect <span class="live-dot">● LIVE</span>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Session state
# -----------------------------
if "comments" not in st.session_state:
    st.session_state.comments = [
        {"text": "Welcome to the stream!", "likes": 2},
        {"text": "Amazing content 🔥", "likes": 4}
    ]

if "likes" not in st.session_state:
    st.session_state.likes = 0

# -----------------------------
# Layout
# -----------------------------
video_col, chat_col = st.columns([3, 1])

# 🎥 VIDEO AREA
with video_col:
    st.image(
        "https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif",
        use_container_width=True
    )

    st.markdown(f"### 👍 Likes: {st.session_state.likes}")

    react_cols = st.columns(4)
    reactions = ["❤️", "👍", "😂", "🔥"]

    for i, r in enumerate(reactions):
        if react_cols[i].button(r, key=f"react_{i}"):
            st.session_state.likes += 1
            st.toast(f"{r} reaction added!")

# 💬 CHAT AREA
with chat_col:
    st.subheader("💬 Live Chat")

    for idx, comment in enumerate(st.session_state.comments):
        st.markdown(f"""
        <div class="chat-box">
        {comment["text"]}<br>
        <span class="comment-like">❤️ {comment["likes"]}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("❤️ Like", key=f"comment_like_{idx}"):
            comment["likes"] += 1
            st.rerun()

# -----------------------------
# COMMENT INPUT
# -----------------------------
st.write("---")

comment_input = st.text_input("Add a comment…")

send_col, react_col = st.columns([5,1])

with send_col:
    if st.button("Send", key="send_comment"):
        if comment_input.strip() == "":
            st.warning("Comment cannot be empty")
        else:
            cleaned = clean_text(comment_input)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]

            if prediction == 1:
                st.error("⚠️ Comment blocked: harmful content detected")
            else:
                st.session_state.comments.append(
                    {"text": comment_input, "likes": 0}
                )
                st.success("✅ Comment posted")
                st.rerun()

with react_col:
    if st.button("❤️", key="bottom_react"):
        st.session_state.likes += 1
        st.rerun()