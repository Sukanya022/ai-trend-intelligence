import streamlit as st
from model import ai_analyst

st.set_page_config(page_title="AI Trend Intelligence", page_icon="🔥", layout="wide")

# ------------------ CSS ------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f172a, #020617);
}

.block-container {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6a00, #ff3c83);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 40px;
}

.glass {
    background: rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 25px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}

.metric-box {
    background: #111827;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.trend-up {
    color: #22c55e;
    font-size: 22px;
    font-weight: bold;
}

.trend-down {
    color: #ef4444;
    font-size: 22px;
    font-weight: bold;
}

.stTextInput > div > div > input {
    background-color: #020617;
    border: 1px solid #ff4b4b;
    border-radius: 10px;
    padding: 12px;
}

.stButton>button {
    background: linear-gradient(90deg, #ff4b4b, #ff6a00);
    border-radius: 12px;
    color: white;
    font-weight: 600;
    padding: 10px 25px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="title">🔥 AI Trend Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced trend detection using AI, ML & NLP</div>', unsafe_allow_html=True)

# ------------------ INPUT ------------------
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input("🔍 Enter topic", placeholder="AI agents, blockchain, startups...")

with col2:
    top_k = st.slider("Top results", 3, 10, 5)

# ------------------ BUTTON ------------------
if st.button("🚀 Analyze Trend"):

    if not query.strip():
        st.warning("Enter a topic first")
    else:
        with st.spinner("Running AI models..."):
            pred, prob, similar, explanation = ai_analyst(query, top_k)

        # ------------------ KPI ROW ------------------
        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            if pred == 1:
                st.markdown('<p class="trend-up">🔥 Trending</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="trend-down">❄️ Not Trending</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Confidence", f"{prob*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with colC:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Results Found", len(similar))
            st.markdown('</div>', unsafe_allow_html=True)

        # ------------------ PROGRESS ------------------
        st.progress(min(int(prob * 100), 100))

        # ------------------ SIMILAR ------------------
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🔗 Similar Trends")

        for s in similar:
            st.markdown(f"• {s}")

        st.markdown('</div>', unsafe_allow_html=True)

        # ------------------ INSIGHT ------------------
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🤖 AI Insight")
        st.write(explanation)
        st.markdown('</div>', unsafe_allow_html=True)

        # ------------------ CHART ------------------
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📈 Trend Strength")

        scores = [len(s.split()) for s in similar]
        st.line_chart(scores)

        st.markdown('</div>', unsafe_allow_html=True)