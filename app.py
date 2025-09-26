from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import streamlit as st
import re
import torch.nn.functional as F

# ==============================
# Config
# ==============================
MODEL_DIR = r"D:\Downloads\final_model_distilbert"
MAX_LEN = 256

st.set_page_config(
    page_title="AI vs Human Text Classifier",
    page_icon="🤖",
    layout="centered",
)

# ==============================
# Load Model & Tokenizer
# ==============================
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

# ==============================
# Preprocessing
# ==============================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs only
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==============================
# Prediction
# ==============================
def predict(text, tokenizer, model, device, threshold=80, margin=20):
    text = clean_text(text)
    if len(text.strip()) == 0:
        return 0.0, 0.0, "Text too short to classify"
    
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    
    human_pct = probs[0] * 100
    ai_pct = probs[1] * 100
    
    if ai_pct >= threshold:
        final_label = "AI Generated Text"
    elif human_pct >= threshold:
        final_label = "Human Written Text"
    elif abs(ai_pct - human_pct) < margin:
        final_label = "Uncertain (Looks close to both)"
    else:
        final_label = "AI Generated Text" if ai_pct > human_pct else "Human Written Text"
    
    return human_pct, ai_pct, final_label

# ==============================
# Streamlit App
# ==============================
def main():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #1e1e2f, #2d2d44);
            color: #f1f1f1;
            font-family: "Segoe UI", sans-serif;
        }
        .title {
            text-align: center;
            color: #f8f9fa;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #dcdcdc;
            font-size: 16px;
            margin-bottom: 30px;
        }
        .footer {
            text-align: center;
            font-size: 13px;
            margin-top: 30px;
            color: #888;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h1 class='title'>🤖 Human Response Identifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Detect whether your text is written by a Human or AI</p>", unsafe_allow_html=True)

    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)

    user_input = st.text_area("✍️ Enter text to classify:", height=150)
    tokenizer, model, device = load_model_and_tokenizer()

    if st.button("🔍 Classify"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to classify.")
        else:
            human_pct, ai_pct, final_label = predict(user_input, tokenizer, model, device)

            # Progress Bars styled as percentages
            st.write("### 🔎 Confidence Levels")
            st.progress(int(human_pct))
            st.write(f"Human Written Probability: **{human_pct:.2f}%**")
            st.progress(int(ai_pct))
            st.write(f"AI Generated Probability: **{ai_pct:.2f}%**")

            # Final prediction styled
            if "AI" in final_label:
                st.error(f"🚨 Final Prediction: {final_label}")
            elif "Human" in final_label:
                st.success(f"✅ Final Prediction: {final_label}")
            else:
                st.warning(f"⚖️ Final Prediction: {final_label}")



if __name__ == "__main__":
    main()
