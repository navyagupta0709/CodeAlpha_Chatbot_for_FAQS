import streamlit as st
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time

# ─── Groq (optional fallback engine) ───────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

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

/* User Message */
.msg-user {
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
    gap: 0.6rem;
    margin: 0.4rem 0;
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
    margin: 0.4rem 0;
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

/* Source tag */
.source-tag {
    display: inline-block;
    border-radius: 6px;
    padding: 0.1rem 0.5rem;
    font-size: 0.68rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.source-faq { background: rgba(34,211,238,0.12); color: #22d3ee; }
.source-ai  { background: rgba(167,139,250,0.12); color: #a78bfa; }

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

# ════════════════════════════════════════════════════════════════════════════
# FAQ KNOWLEDGE BASE (embedded directly — no separate file needed)
# ════════════════════════════════════════════════════════════════════════════
faqs = [
    {"question": "What is artificial intelligence?",
     "answer": "Artificial Intelligence (AI) is the simulation of human intelligence in machines programmed to think, learn, and problem-solve like humans. It includes subfields like machine learning, deep learning, and natural language processing."},
    {"question": "What is machine learning?",
     "answer": "Machine Learning (ML) is a subset of AI where algorithms learn patterns from data without being explicitly programmed. It improves automatically through experience using methods like supervised, unsupervised, and reinforcement learning."},
    {"question": "What is deep learning?",
     "answer": "Deep Learning is a subset of ML using neural networks with many layers (deep neural networks). It excels at tasks like image recognition, speech processing, and natural language understanding."},
    {"question": "What is natural language processing?",
     "answer": "Natural Language Processing (NLP) is the branch of AI focused on enabling computers to understand, interpret, and generate human language. Applications include chatbots, translation, sentiment analysis, and text summarization."},
    {"question": "What is a neural network?",
     "answer": "A neural network is a computational model inspired by the human brain. It consists of layers of interconnected nodes (neurons) that process data and learn to recognize patterns by adjusting connection weights during training."},
    {"question": "What is supervised learning?",
     "answer": "Supervised learning is a type of ML where the model is trained on labeled data — input-output pairs — so it learns to map inputs to correct outputs. Examples include classification and regression tasks."},
    {"question": "What is unsupervised learning?",
     "answer": "Unsupervised learning trains on unlabeled data to find hidden patterns or structure. Common techniques include clustering (e.g., K-Means) and dimensionality reduction (e.g., PCA)."},
    {"question": "What is reinforcement learning?",
     "answer": "Reinforcement Learning (RL) is an ML paradigm where an agent learns by interacting with an environment, receiving rewards or penalties. It powers applications like game-playing AIs (AlphaGo) and robotics."},
    {"question": "What is overfitting in machine learning?",
     "answer": "Overfitting occurs when a model learns the training data too well — including noise — and performs poorly on new, unseen data. It's prevented using techniques like regularization, dropout, and cross-validation."},
    {"question": "What is a large language model?",
     "answer": "A Large Language Model (LLM) is a deep learning model trained on massive text datasets to understand and generate human-like text. Examples include GPT-4, Claude, and LLaMA."},
    {"question": "What is transfer learning?",
     "answer": "Transfer learning involves taking a pre-trained model (e.g., BERT, ResNet) and fine-tuning it for a new, related task. It saves training time and works well when labeled data is limited."},
    {"question": "What is a convolutional neural network?",
     "answer": "A Convolutional Neural Network (CNN) is a deep learning architecture designed for processing structured grid data like images. It uses convolutional layers to detect features like edges, textures, and objects."},
    {"question": "What is the difference between AI and ML?",
     "answer": "AI is the broad concept of creating intelligent machines, while ML is a specific approach to achieve AI using data-driven learning. All ML is AI, but not all AI is ML — rule-based systems are AI without ML."},
    {"question": "What is Python used for in AI?",
     "answer": "Python is the dominant language in AI/ML due to its simplicity and rich ecosystem: NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch, and Keras. It's used for data preprocessing, model building, and deployment."},
    {"question": "What is a random forest?",
     "answer": "Random Forest is an ensemble learning method that builds multiple decision trees and merges their predictions for better accuracy and robustness. It's widely used for classification and regression tasks."},
    {"question": "What is a support vector machine?",
     "answer": "Support Vector Machine (SVM) is a supervised ML algorithm that finds the optimal hyperplane to separate classes in feature space. It works well for high-dimensional data and binary classification."},
    {"question": "What is feature engineering?",
     "answer": "Feature engineering is the process of selecting, transforming, or creating input variables (features) from raw data to improve model performance. It requires domain knowledge and creativity."},
    {"question": "What is a generative adversarial network?",
     "answer": "A GAN consists of two neural networks — a generator and a discriminator — competing against each other. The generator creates fake data; the discriminator tries to detect it. Used in image synthesis, deepfakes, and data augmentation."},
    {"question": "What is computer vision?",
     "answer": "Computer Vision enables machines to interpret and understand visual information from images or videos. Applications include facial recognition, medical imaging, self-driving cars, and object detection."},
    {"question": "What is the bias-variance tradeoff?",
     "answer": "Bias refers to errors from overly simple models; variance refers to errors from overly complex models. The tradeoff is finding the right complexity — low bias and low variance — to achieve good generalization."},
    {"question": "What is data preprocessing?",
     "answer": "Data preprocessing involves cleaning and transforming raw data before feeding it into an ML model. Steps include handling missing values, encoding categorical variables, normalization, and splitting into train/test sets."},
    {"question": "What is a transformer model?",
     "answer": "Transformers are deep learning architectures using self-attention mechanisms to process sequential data in parallel. Introduced in 'Attention is All You Need' (2017), they power most modern NLP models like BERT and GPT."},
    {"question": "What is gradient descent?",
     "answer": "Gradient descent is an optimization algorithm used to minimize the loss function in ML models by iteratively adjusting parameters in the direction of the steepest descent (negative gradient)."},
    {"question": "What are hyperparameters?",
     "answer": "Hyperparameters are configuration settings set before training (e.g., learning rate, batch size, number of layers). Unlike model parameters, they are not learned from data and must be tuned manually or via grid/random search."},
    {"question": "What is YOLO in object detection?",
     "answer": "YOLO (You Only Look Once) is a real-time object detection algorithm that processes the entire image in a single pass through a neural network, predicting bounding boxes and class probabilities simultaneously."},
]

# ─── Sidebar Settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Free key from console.groq.com — used only as a fallback when no FAQ matches confidently."
    )

    enable_fallback = st.toggle(
        "Enable AI Fallback",
        value=True,
        help="When ON, questions outside the FAQ list get answered by an LLM (LLaMA 3.3 70B via Groq) instead of 'no match found'."
    )

    st.markdown("---")
    st.caption(
        "🎯 **FAQ Match** = exact answer from our curated knowledge base (fast, always accurate).\n\n"
        "🧠 **AI Fallback** = general-purpose answer generated live by an LLM "
        "when your question isn't in the FAQ list."
    )

# ─── NLP Engine (preprocessing + TF-IDF + cosine similarity) ──────────────────
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


def get_faq_answer(user_query, threshold=0.15):
    """Try to match the query against our curated FAQ list using TF-IDF + cosine similarity."""
    processed_q = preprocess(user_query)
    query_vec = vectorizer.transform([processed_q])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score < threshold:
        return None, best_score, None

    return faqs[best_idx]["answer"], best_score, faqs[best_idx]["question"]


def get_ai_fallback_answer(user_query, api_key):
    """
    Ask an LLM (LLaMA 3.3 70B via Groq) for a general answer when the FAQ
    list doesn't have a confident match. Returns None if no key is set or
    the API call fails for any reason.
    """
    if not api_key or not GROQ_AVAILABLE:
        return None

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are AskAI, a friendly and knowledgeable assistant. "
                        "Answer the user's question clearly and concisely in 2-4 sentences."
                    ),
                },
                {"role": "user", "content": user_query},
            ],
            temperature=0.5,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def answer_query(user_query, fallback_enabled, api_key):
    """
    Main routing function: try the FAQ match first (this is the technique
    required by the task spec). If it's not confident AND fallback is
    enabled, ask the LLM instead of giving up.

    Returns: (answer_text, source, score_or_None, matched_question_or_None)
    source is one of: "faq", "ai", "none"
    """
    faq_answer, score, matched_question = get_faq_answer(user_query)

    if faq_answer:
        return faq_answer, "faq", score, matched_question

    if fallback_enabled:
        ai_answer = get_ai_fallback_answer(user_query, api_key)
        if ai_answer:
            return ai_answer, "ai", score, None

    return (
        "I don't have a confident answer for that. Try rephrasing, "
        "or enable AI Fallback in the sidebar with a Groq API key for general questions.",
        "none",
        score,
        None,
    )


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">NLP · TF-IDF · COSINE SIMILARITY + AI FALLBACK</div>
    <h1>Ask<span class="accent">AI</span></h1>
    <p>Your intelligent FAQ assistant — now with AI fallback for anything else</p>
</div>
""", unsafe_allow_html=True)

# ─── Init session ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0


def render_bot_message(text, source, score, matched):
    conf_pct = int((score or 0) * 100)

    if source == "faq":
        tag_html = '<span class="source-tag source-faq">🎯 FAQ MATCH</span>'
        meta = f"Confidence: {conf_pct}% · Matched: <em>{matched}</em>"
        low_conf_html = ""
    elif source == "ai":
        tag_html = '<span class="source-tag source-ai">🧠 AI GENERATED</span>'
        meta = "Answered by LLaMA 3.3 70B (no FAQ match found)"
        low_conf_html = ""
    else:
        tag_html = ""
        meta = f"Confidence: {conf_pct}% (below threshold)"
        low_conf_html = '<div class="low-conf">⚠️ No confident match — enable AI Fallback for general questions</div>'

    st.markdown(f"""
    <div class="msg-bot">
        <div class="avatar-bot">🤖</div>
        <div class="bubble-bot">
            {tag_html}<br>
            {text}
            {low_conf_html}
            <div class="confidence-bar">{meta}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Welcome / Suggestions ────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-box">
        👋 Hi! I'm <strong>AskAI</strong>.<br>
        Ask me about <strong>Machine Learning, Deep Learning, NLP</strong> and more — I'll pull
        the exact answer from my FAQ database.<br>
        Ask something outside that scope, and (if AI Fallback is on) I'll still try my best!
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        "What is machine learning?",
        "Explain deep learning",
        "What's the weather like today?",
        "What is overfitting?",
    ]
    cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        if cols[i].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": sug})
            answer, source, score, matched = answer_query(sug, enable_fallback, groq_api_key)
            st.session_state.messages.append({
                "role": "bot", "content": answer,
                "source": source, "score": score, "matched": matched
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
        render_bot_message(msg['content'], msg.get('source', 'none'), msg.get('score', 0), msg.get('matched'))

# ─── Input ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
col_in, col_send = st.columns([5, 1])
with col_in:
    user_input = st.text_input(
        "",
        placeholder="Ask me anything...",
        key=f"user_input_{st.session_state.input_key}",
        label_visibility="collapsed"
    )
with col_send:
    send = st.button("Send ➤", use_container_width=True, type="primary")

if send and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    with st.spinner("Thinking..."):
        time.sleep(0.2)
        answer, source, score, matched = answer_query(user_input.strip(), enable_fallback, groq_api_key)
    st.session_state.messages.append({
        "role": "bot", "content": answer,
        "source": source, "score": score, "matched": matched
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
    Built with ❤️ by Navya · CodeAlpha AI Internship · Task 2 — FAQ Chatbot · TF-IDF + Cosine Similarity + AI Fallback
</div>
""", unsafe_allow_html=True)
