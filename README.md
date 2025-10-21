# 📧 Email Spam Detection App

A web application to detect whether an email is **spam** or **not spam**.  
Built with **Python**, **Streamlit**, and **Machine Learning**, with optional AI explanations using **Google Gemini API**.

---

## 🔍 Features

- Detects spam emails using a **Random Forest / Logistic Regression classifier**.
- Preprocesses emails by:
  - Converting to lowercase
  - Removing unnecessary punctuation
  - Handling common spam phrases
- Optional: Provides **AI-generated explanations** for why an email is considered spam (via Google Gemini API).
- Interactive **Streamlit interface** — input any email and get instant results.

---

## 🧰 Tech Stack

- Python 3.10+
- Streamlit
- scikit-learn
- pandas, numpy
- joblib
- google-generativeai (optional, for AI explanations)
- TF-IDF vectorizer for feature extraction

---

## 🚀 Installation

1. Clone this repository:

```bash
git clone https://github.com/NikhilBansalv/email-spam-detector.git
cd email-spam-detector
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the streamlit app

```bash
streamlit run app.py
```

## ⚠️ Note:

Replace your api key before running on the local machine.
