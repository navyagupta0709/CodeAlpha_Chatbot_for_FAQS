# 🤖 AskAI — FAQ Chatbot
> CodeAlpha AI Internship | Task 2

An NLP-powered FAQ chatbot that answers Artificial Intelligence questions using TF-IDF vectorization and cosine similarity matching.

## ✨ Features
- 🧠 NLP preprocessing: tokenization, stopword removal, lemmatization (NLTK)
- 📊 TF-IDF vectorization with bigrams
- 🎯 Cosine similarity-based intent matching
- 💬 Chat UI with confidence scores and matched FAQ display
- ⚡ Suggestion chips for quick-start
- ⚠️ Low confidence warnings

## 🛠️ Tech Stack
| Layer | Tech |
|-------|------|
| Frontend | Streamlit |
| NLP | NLTK (tokenization, lemmatization, stopwords) |
| Vectorization | Scikit-learn TF-IDF |
| Similarity | Cosine Similarity |
| Deployment | AWS Elastic Beanstalk |
| Language | Python 3.10+ |

## 🚀 Run Locally

```bash
git clone https://github.com/navyagupta0709/CodeAlpha_FAQChatbot
cd CodeAlpha_FAQChatbot
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Repo Structure
```
CodeAlpha_FAQChatbot/
├── app.py
├── faq_data.py
├── requirements.txt
└── README.md
```

---
Made with ❤️ by **Navya** | B.Tech ECE (AI/ML) | Punjabi University, Patiala
