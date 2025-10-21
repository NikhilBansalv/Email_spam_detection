import streamlit as st
import joblib
import re
import google.generativeai as genai
import os

# Configure Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'^subject\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-zA-Z0-9\s!$%]', '', text)
    return text

def gemini_explain_spam(email_text):
    model_gemini = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    Analyze this email and explain briefly (2–3 sentences) why it looks like spam:
    ---
    {email_text}
    ---
    Keep it clear, helpful, and professional.
    """
    response = model_gemini.generate_content(prompt)
    return response.text.strip()

st.set_page_config(page_title="📧 Spam Detector", page_icon="🚫")
st.title("📧 Email Spam Detection App")

email_text = st.text_area("✉️ Enter email text:", height=200)

if st.button("🔍 Check Spam"):
    if email_text.strip() == "":
        st.warning("Please enter an email to analyze.")
    else:
        cleaned = clean_text(email_text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.error("🚫 This email is **SPAM**!")
            with st.spinner("Analyzing why it might be spam..."):
                reason = gemini_explain_spam(email_text)
            st.write("### 🧠 Why it looks like spam:")
            st.info(reason)
        else:
            st.success("✅ This email is **NOT SPAM**!")
