import streamlit as st
import joblib
import re
import emoji
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from streamlit_webrtc import webrtc_streamer
import random
import math

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

bad_words = ["stupid","idiot","hate","die","loser","ugly","dumb"]

# ---------------- TEXT CLEANING ----------------
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
        score = model.predict_proba(vec).ravel()[1]

    elif hasattr(model, "decision_function"):
        score = model.decision_function(vec).ravel()[0]
        score = 1 / (1 + math.exp(-score))

    else:
        score = model.predict(vec).ravel()[0]

    return float(score)

# ---------------- SENTIMENT ----------------
def sentiment(text):
    if any(w in text.lower() for w in bad_words):
        return "😠 Negative"
    if any(w in text.lower() for w in ["love","great","nice","awesome","good"]):
        return "🙂 Positive"
    return "😐 Neutral"

def highlight_bad_words(text):
    for w in bad_words:
        text = re.sub(fr"\b{w}\b",
                      f"<span style='color:red'><b>{w}</b></span>",
                      text,
                      flags=re.I)
    return text

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- DARK MODE ----------------
dark_mode = st.toggle("🌙 Dark Mode", value=True)

bg = "#0f172a" if dark_mode else "#f8fafc"
text_color = "#f1f5f9" if dark_mode else "#0f172a"
card = "#1e293b" if dark_mode else "#ffffff"

st.markdown(f"""
<style>
.stApp {{ background:{bg}; color:{text_color}; }}

.chat-card {{
    background:{card};
    padding:10px;
    border-radius:12px;
    margin-bottom:10px;
    display:flex;
    gap:10px;
}}

.overlay {{
    position: fixed;
    bottom: 120px;
    left: 40%;
    pointer-events: none;
}}

.float {{
    position: absolute;
    font-size: 28px;
    animation: floatUp 3s linear forwards;
}}

@keyframes floatUp {{
    0% {{ transform: translateY(0); opacity:1; }}
    100% {{ transform: translateY(-300px); opacity:0; }}
}}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
left, right = st.columns([6,1])

with left:
    st.title("📺 StreamConnect ✔️")

with right:
    if "live" not in st.session_state:
        st.session_state.live = False

    if st.session_state.live:
        if st.button("⏹ Stop"):
            st.session_state.live = False
            st.rerun()
    else:
        if st.button("▶ Start"):
            st.session_state.live = True
            st.rerun()

# FIXED display (prevents help dump)
if st.session_state.live:
    st.success("🔴 LIVE")
else:
    st.info("Stream offline")

# ---------------- USERS ----------------
users = {
    "Alex":{"avatar":"https://i.pravatar.cc/40?img=1","verified":True},
    "Priya":{"avatar":"https://i.pravatar.cc/40?img=5","verified":False},
    "You":{"avatar":"https://i.pravatar.cc/40?img=12","verified":False}
}

# ---------------- SESSION STATE ----------------
for key, default in {
    "comments": [],
    "floating": [],
    "pending": None,
    "warnings": 0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

video, chat = st.columns([3,1])

# 🎥 VIDEO
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
        verified = " ✔️" if user["verified"] else ""

        st.markdown(f"""
        <div class="chat-card">
            <img src="{user['avatar']}" width="40" style="border-radius:50%">
            <div>
                <b>{c['user']}{verified}</b><br>
                {highlight_bad_words(c['text'])}
            </div>
        </div>
        """, unsafe_allow_html=True)

# FLOATING EMOJIS
overlay = '<div class="overlay">'
for e in st.session_state.floating[-6:]:
    left = random.randint(0,200)
    overlay += f'<div class="float" style="left:{left}px;">{e}</div>'
overlay += '</div>'
st.markdown(overlay, unsafe_allow_html=True)

# ---------------- COMMENT INPUT ----------------
st.write("---")
comment = st.text_input("Add a comment", disabled=not st.session_state.live)

if comment:
    st.caption(f"Sentiment: {sentiment(comment)}")

if st.button("Send", disabled=not st.session_state.live):

    tox = toxicity_score(clean_text(comment))
    percent = int(tox*100)

    st.progress(percent)
    st.write(f"Toxicity: {percent}%")

    if tox < 0.40:
        st.session_state.comments.append({"user":"You","text":comment})
        st.success("Posted")

    elif tox < 0.70:
        st.warning("⚠️ Consider editing")
        st.session_state.pending = comment

    else:
        st.error("🚫 Harmful message blocked")
        st.session_state.warnings += 1

# EDIT FLOW
if st.session_state.pending:
    st.info("Edit your message?")
    if st.button("Edit"):
        st.session_state.pending = None
        st.rerun()

st.write(f"⚠️ Warnings: {st.session_state.warnings}")