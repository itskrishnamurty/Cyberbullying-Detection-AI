import streamlit as st
import joblib
import re
import emoji
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from streamlit_webrtc import webrtc_streamer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
import math

# ---------------- LOAD MODELS ----------------
model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

sentiment_ai = SentimentIntensityAnalyzer()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------- USERS ----------------
users = {
    "You": {"avatar": "https://i.pravatar.cc/40?img=12", "verified": False},
    "Alex": {"avatar": "https://i.pravatar.cc/40?img=1", "verified": True},
    "Priya": {"avatar": "https://i.pravatar.cc/40?img=5", "verified": False},
}

# ---------------- TEXT CLEAN ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- TOXICITY SCORE ----------------
def toxicity_score(text):
    vec = vectorizer.transform([text])

    if hasattr(model, "predict_proba"):
        ml_score = model.predict_proba(vec).ravel()[1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(vec).ravel()[0]
        ml_score = 1 / (1 + math.exp(-s))
    else:
        ml_score = model.predict(vec).ravel()[0]

    sentiment_score = sentiment_ai.polarity_scores(text)["compound"]
    aggression_boost = abs(sentiment_score) * 0.25 if sentiment_score < 0 else 0

    return max(0, min(1, ml_score + aggression_boost))

# ---------------- SENTIMENT ----------------
def sentiment(text):
    score = sentiment_ai.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "🙂 Positive"
    elif score <= -0.05:
        return "😠 Negative"
    else:
        return "😐 Neutral"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- DARK MODE ----------------
dark_mode = st.toggle("🌙 Dark Mode", value=True)

bg = "#0f172a" if dark_mode else "#f4f6f8"
text_color = "#f1f5f9" if dark_mode else "#0f172a"
card = "#1e293b" if dark_mode else "#ffffff"

# ---------------- GLOBAL CSS + HEADING ----------------
st.markdown(f"""
<style>
.stApp {{ background:{bg}; color:{text_color}; }}

.main-title {{
    text-align:center;
    font-size:64px;
    font-weight:800;
    letter-spacing:2px;
    background: linear-gradient(90deg,#ff4ecd,#7c7cff,#00e0ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation: glow 3s ease-in-out infinite alternate;
    transition: transform .3s ease;
}}
.main-title:hover {{ transform: scale(1.05); }}

@keyframes glow {{
    from {{ text-shadow:0 0 10px rgba(255,78,205,.6); }}
    to {{ text-shadow:0 0 25px rgba(0,224,255,.9); }}
}}

.subtitle {{
    text-align:center;
    font-size:16px;
    opacity:.75;
    margin-top:-10px;
}}

.chat-card {{
    background:{card};
    padding:10px;
    border-radius:12px;
    margin-bottom:10px;
    display:flex;
    gap:10px;
}}

.safe {{ border-left:5px solid #22c55e; }}
.warn {{ border-left:5px solid #f59e0b; }}
.block {{ border-left:5px solid #ef4444; }}

.overlay {{ position:fixed; bottom:120px; left:40%; }}
.float {{
    position:absolute;
    font-size:28px;
    animation: floatUp 3s linear forwards;
}}

@keyframes floatUp {{
    0% {{ transform:translateY(0); opacity:1; }}
    100% {{ transform:translateY(-300px); opacity:0; }}
}}
</style>

<div class="main-title">Cyberbullying Detection</div>
<div class="subtitle">AI Moderated Social comment detector Platform</div>
""", unsafe_allow_html=True)

# ---------------- LIVE TOGGLE ----------------
if "live" not in st.session_state:
    st.session_state.live = False

col1, col2 = st.columns([6,1])

with col2:
    if st.session_state.live:
        if st.button("⏹ Stop"):
            st.session_state.live = False
            st.rerun()
    else:
        if st.button("▶ Start"):
            st.session_state.live = True
            st.rerun()

if st.session_state.live:
    st.success("🔴 LIVE")
else:
    st.info("Stream offline")

# ---------------- SESSION STATE ----------------
for key, default in {
    "comments": [],
    "floating": [],
    "warnings": 0,
    "last_score": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

video, chat = st.columns([3,1])

# 🎥 CAMERA
with video:
    st.subheader("Live Camera")
    if st.session_state.live:
        webrtc_streamer(key="live_stream")
    else:
        st.warning("Camera off")

    reacts = ["❤️","👍","😂","🔥"]
    cols = st.columns(4)
    for i,r in enumerate(reacts):
        if cols[i].button(r, key=f"react{i}", disabled=not st.session_state.live):
            st.session_state.floating.append(r)
            st.rerun()

# 💬 CHAT
with chat:
    st.subheader("Live Chat")
    for c in st.session_state.comments:
        user = users[c["user"]]
        st.markdown(f"""
        <div class="chat-card {c['class']}">
            <img src="{user['avatar']}" width="40" style="border-radius:50%">
            <div>
                <b>{c['user']}</b> • {c['time']}<br>
                {c['text']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# FLOATING EMOJIS
overlay = '<div class="overlay">'
for e in st.session_state.floating[-6:]:
    overlay += f'<div class="float" style="left:{random.randint(0,200)}px;">{e}</div>'
overlay += '</div>'
st.markdown(overlay, unsafe_allow_html=True)

# ---------------- COMMENT INPUT ----------------
st.write("---")
comment = st.text_input("Add a comment", disabled=not st.session_state.live)

if comment:
    st.caption(f"Sentiment: {sentiment(comment)}")

if st.button("Send", disabled=not st.session_state.live):

    score = toxicity_score(clean_text(comment))
    percent = int(score * 100)
    st.session_state.last_score = percent

    if score <= 0.01:
        css = "safe"
        st.success("Posted")

    elif score < 0.03:
        css = "warn"
        st.warning("⚠️ Consider editing")

    else:
        css = "block"
        st.error("🚫 Harmful message blocked")
        st.session_state.warnings += 1

    if css != "block":
        st.session_state.comments.append({
            "user": "You",
            "text": comment,
            "time": datetime.now().strftime("%H:%M"),
            "class": css
        })

    st.rerun()

# SHOW METER AFTER RERUN
if st.session_state.last_score is not None:
    st.progress(st.session_state.last_score)
    st.write(f"Toxicity Level: {st.session_state.last_score}%")

st.write(f"⚠️ Warnings: {st.session_state.warnings}")