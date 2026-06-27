import streamlit as st
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
from faq_data import faqs

# ─── Download NLTK data ────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AskAI — FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; min-height: 100vh; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 780px; }

/* Hero */
.hero {
    text-align: center;
    padding: 2rem 1rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.3rem;
}
.hero h1 .accent { color: #22d3ee; }
.hero p { color: #64748b; font-size: 0.9rem; }
.badge {
    display: inline-block;
    background: rgba(34, 211, 238, 0.1);
    color: #22d3ee;
    border: 1px solid rgba(34, 211, 238, 0.2);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    margin-bottom: 0.8rem;
}

/* Chat Container */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1rem;
}

/* User Message */
.msg-user {
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
    gap: 0.6rem;
}
.bubble-user {
    background: linear-gradient(135deg, #0ea5e9, #22d3ee);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.5;
    box-shadow: 0 2px 12px rgba(14,165,233,0.25);
}
.avatar-user {
    width: 32px; height: 32px;
    background: #0ea5e9;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-top: 2px;
}

/* Bot Message */
.msg-bot {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 0.6rem;
}
.bubble-bot {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 0.75rem 1.1rem;
    max-width: 80%;
    font-size: 0.92rem;
    line-height: 1.6;
}
.avatar-bot {
    width: 32px; height: 32px;
    background: rgba(34,211,238,0.15);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.confidence-bar {
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    font-size: 0.72rem;
    color: #64748b;
}
.conf-fill {
    display: inline-block;
    height: 3px;
    border-radius: 2px;
    background: #22d3ee;
    vertical-align: middle;
    margin: 0 0.4rem;
}

/* Welcome message */
.welcome-box {
    background: rgba(34,211,238,0.05);
    border: 1px solid rgba(34,211,238,0.15);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}
.welcome-box strong { color: #22d3ee; }

/* Suggestion chips */
.chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.2rem; }
.chip {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 0.3rem 0.85rem;
    color: #94a3b8;
    font-size: 0.77rem;
    cursor: pointer;
}

/* Low confidence */
.low-conf {
    background: rgba(251,146,60,0.08);
    border: 1px solid rgba(251,146,60,0.2);
    color: #fb923c;
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    font-size: 0.82rem;
    margin-top: 0.4rem;
}

/* Footer */
.footer { text-align:center; color: #334155; font-size:0.73rem; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

# ─── NLP Pipeline ──────────────────────────────────────────────────────────────
@st.cache_resource
def build_nlp_engine():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
        return ' '.join(tokens)

    questions = [f["question"] for f in faqs]
    processed = [preprocess(q) for q in questions]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(processed)

    return vectorizer, tfidf_matrix, preprocess

vectorizer, tfidf_matrix, preprocess = build_nlp_engine()

def get_answer(user_query, threshold=0.15):
    processed_q = preprocess(user_query)
    query_vec = vectorizer.transform([processed_q])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score < threshold:
        return None, best_score, None

    return faqs[best_idx]["answer"], best_score, faqs[best_idx]["question"]

# ─── Init session ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">NLP POWERED · COSINE SIMILARITY</div>
    <h1>Ask<span class="accent">AI</span></h1>
    <p>Your intelligent FAQ assistant for Artificial Intelligence topics</p>
</div>
""", unsafe_allow_html=True)

# ─── Welcome / Suggestions ────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-box">
        👋 Hi! I'm <strong>AskAI</strong>, your AI knowledge assistant.<br>
        I can answer questions on <strong>Machine Learning, Deep Learning, NLP, Neural Networks</strong> and more.<br>
        Try one of the suggestions below or type your own question!
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        "What is machine learning?",
        "Explain deep learning",
        "What is a neural network?",
        "What is overfitting?",
        "How does YOLO work?",
    ]
    cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        if cols[i].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": sug})
            answer, score, matched_q = get_answer(sug)
            if answer:
                st.session_state.messages.append({
                    "role": "bot", "content": answer,
                    "score": score, "matched": matched_q
                })
            else:
                st.session_state.messages.append({
                    "role": "bot",
                    "content": "I couldn't find a confident match. Try rephrasing your question.",
                    "score": score, "matched": None
                })
            st.rerun()

# ─── Chat History ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="bubble-user">{msg['content']}</div>
            <div class="avatar-user">👤</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        conf_pct = int(msg.get('score', 0) * 100)
        bar_width = min(conf_pct * 1.5, 100)
        matched_text = f"Matched: <em>{msg.get('matched', '')}</em>" if msg.get('matched') else ""
        low_conf_html = ""
        if msg.get('score', 1) < 0.3 and msg.get('matched'):
            low_conf_html = '<div class="low-conf">⚠️ Low confidence match — consider rephrasing</div>'

        st.markdown(f"""
        <div class="msg-bot">
            <div class="avatar-bot">🤖</div>
            <div class="bubble-bot">
                {msg['content']}
                {low_conf_html}
                <div class="confidence-bar">
                    Confidence <span class="conf-fill" style="width:{bar_width}px"></span> {conf_pct}%
                    &nbsp;·&nbsp; {matched_text}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Input ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
col_in, col_send = st.columns([5, 1])
with col_in:
    user_input = st.text_input(
        "",
        placeholder="Ask me anything about AI, ML, Deep Learning...",
        key=f"user_input_{st.session_state.input_key}",
        label_visibility="collapsed"
    )
with col_send:
    send = st.button("Send ➤", use_container_width=True, type="primary")

if send and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    with st.spinner("Thinking..."):
        time.sleep(0.3)
        answer, score, matched_q = get_answer(user_input.strip())
    if answer:
        st.session_state.messages.append({
            "role": "bot", "content": answer,
            "score": score, "matched": matched_q
        })
    else:
        st.session_state.messages.append({
            "role": "bot",
            "content": "I'm not confident about that one. Please try rephrasing or ask something else about AI/ML.",
            "score": score, "matched": None
        })
    st.session_state.input_key += 1
    st.rerun()

# ─── Clear Chat ───────────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Clear Chat", use_container_width=False):
        st.session_state.messages = []
        st.session_state.input_key += 1
        st.rerun()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ by Navya · CodeAlpha AI Internship · Task 2 — FAQ Chatbot · NLP + Cosine Similarity
</div>
""", unsafe_allow_html=True)
