import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

SAMPLE_REAL = (
    "The U.S. Senate on Tuesday voted to approve a spending bill that would fund federal "
    "agencies through the end of the fiscal year, sending the measure to the president for "
    "his signature. The vote was 79 to 18, with several Republicans and Democrats crossing "
    "party lines. Lawmakers said the compromise followed weeks of negotiations over defense "
    "spending and border security funding, according to a statement from the Senate majority "
    "leader's office."
)
SAMPLE_FAKE = (
    "SHOCKING!!! You won't BELIEVE what the corrupt media is hiding from you! Insiders reveal "
    "the SECRET plan that elites don't want you to know about. Share this before it gets "
    "DELETED! Wake up America, they are LYING to you and the mainstream media will never "
    "report this bombshell truth!!!"
)

# ---------- STYLE ----------
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 820px;}
.hero {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); border-radius: 18px; padding: 26px 30px; display: flex; align-items: center; gap: 22px; margin-bottom: 18px;}
.hero h1 {color: #ffffff; margin: 0; font-size: 2rem; line-height: 1.2; padding: 0;}
.hero p {color: #cfe3ea; margin: 8px 0 0 0; font-size: 1rem;}
.steps {display: flex; gap: 12px; margin: 6px 0 22px 0; flex-wrap: wrap;}
.step {flex: 1; min-width: 150px; background: rgba(120,140,160,0.12); border: 1px solid rgba(120,140,160,0.28); border-radius: 14px; padding: 14px 16px; font-size: 0.92rem;}
.step b {display: block; margin-bottom: 4px;}
.result {border-radius: 16px; padding: 22px 26px; margin-top: 16px; color: #ffffff;}
.result.real {background: linear-gradient(135deg,#0f766e,#16a34a);}
.result.fake {background: linear-gradient(135deg,#9f1239,#dc2626);}
.result h2 {margin: 0; color: #ffffff; font-size: 1.8rem; padding: 0;}
.result p {margin: 6px 0 0 0; color: #ffffff; opacity: 0.95;}
.bar {height: 12px; border-radius: 8px; background: rgba(255,255,255,0.3); margin-top: 14px; overflow: hidden;}
.bar div {height: 100%; background: #ffffff; border-radius: 8px;}
.note {font-size: 0.85rem; opacity: 0.75; margin-top: 26px;}
</style>""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
if not os.path.exists("model.joblib"):
    st.error("model.joblib not found. Run `python3 train.py` first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()
best_name = open("best_model.txt").read() if os.path.exists("best_model.txt") else "ML model"

# ---------- HEADER BANNER ----------
st.markdown(
    '<div class="hero">'
    '<svg width="84" height="84" viewBox="0 0 84 84" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="8" y="10" width="52" height="62" rx="6" fill="#ffffff"/>'
    '<rect x="15" y="18" width="38" height="7" rx="2" fill="#203a43"/>'
    '<rect x="15" y="31" width="38" height="4" rx="2" fill="#9db4bd"/>'
    '<rect x="15" y="40" width="38" height="4" rx="2" fill="#9db4bd"/>'
    '<rect x="15" y="49" width="24" height="4" rx="2" fill="#9db4bd"/>'
    '<circle cx="56" cy="54" r="15" fill="#2c5364" stroke="#5eead4" stroke-width="5"/>'
    '<line x1="67" y1="65" x2="78" y2="76" stroke="#5eead4" stroke-width="7" stroke-linecap="round"/>'
    '</svg>'
    '<div><h1>Real-Time Fake News Detector</h1>'
    '<p>Paste a news article and a machine learning model tells you if it looks Fake or Real.</p></div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="steps">'
    '<div class="step"><b>1. Paste</b>Copy a news headline or article into the box.</div>'
    '<div class="step"><b>2. Analyse</b>The text is cleaned and converted to numbers (TF-IDF).</div>'
    '<div class="step"><b>3. Result</b>The model predicts Fake or Real with a confidence score.</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------- BUTTON ACTIONS ----------
def load_real():
    st.session_state.news_text = SAMPLE_REAL

def load_fake():
    st.session_state.news_text = SAMPLE_FAKE

def clear_text():
    st.session_state.news_text = ""

# ---------- TABS ----------
tab_detect, tab_results = st.tabs(["🔎 Detector", "📊 Model results"])

with tab_detect:
    c1, c2, c3 = st.columns(3)
    c1.button("📄 Real-style example", on_click=load_real, use_container_width=True)
    c2.button("🚨 Fake-style example", on_click=load_fake, use_container_width=True)
    c3.button("🧹 Clear", on_click=clear_text, use_container_width=True)

    text = st.text_area(
        "News text", key="news_text", height=220,
        placeholder="Paste news content here...",
    )
    st.caption(f"{len(text.split())} words")

    if st.button("🔍 Analyse news", type="primary", use_container_width=True):
        if len(text.strip()) < 30:
            st.warning("Please enter a longer text (at least a couple of sentences).")
        else:
            proba = model.predict_proba([text])[0]   # [P(fake), P(real)]
            is_real = proba[1] >= 0.5
            confidence = max(proba) * 100
            label = "REAL" if is_real else "FAKE"
            css = "real" if is_real else "fake"
            icon = "✅" if is_real else "🚨"
            st.markdown(
                f'<div class="result {css}">'
                f'<h2>{icon} Prediction: {label}</h2>'
                f'<p>Confidence: {confidence:.1f}%</p>'
                f'<div class="bar"><div style="width:{confidence:.1f}%"></div></div>'
                f'<p>Probability real: {proba[1]*100:.1f}% &nbsp;|&nbsp; '
                f'Probability fake: {proba[0]*100:.1f}%</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

with tab_results:
    st.subheader("How well do the models perform?")
    st.write(
        "Three machine learning models were trained and tested on the same data "
        "(80% training, 20% testing)."
    )
    if os.path.exists("results.csv"):
        res = pd.read_csv("results.csv")
        st.dataframe(res, hide_index=True, use_container_width=True)
        st.bar_chart(res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]])
        st.success(f"Best model (highest F1-score): **{best_name}**. This is the model used by the detector.")
    else:
        st.info("Run `python3 train.py` to create the results table.")

st.markdown(
    '<div class="note">⚠️ Learning project. The model was trained on mostly US political news '
    'from 2016-2017 and cannot verify facts. Do not use it for real decisions.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About")
    st.write("Final Year Project: Development of a Real-Time Fake News Detection System Utilizing Machine Learning.")
    st.write(f"**Model in use:** {best_name}")
    st.write("**Method:** TF-IDF features + supervised classification.")
