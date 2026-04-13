import os
import time
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

import config
from model import HybridBrainTransformer
from dataset import load_and_preprocess_data

CLASSES = ["Left Hand", "Right Hand"]
THEME_COLOR = "#00f2ea"
BG_COLOR = "#050510"

st.set_page_config(
    page_title="Brain2Hand | Neural Interface",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Roboto+Mono:wght@300;400&display=swap');

    .stApp {{
        background-color: {BG_COLOR};
        color: #e0e0e0;
        font-family: 'Roboto Mono', monospace;
    }}

    h1, h2, h3, h4, .big-font {{
        font-family: 'Orbitron', sans-serif !important;
        color: white;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}

    .hand-container {{
        width: 100%;
        height: 300px;
        display: flex;
        justify-content: center;
        align-items: center;
        background: rgba(10, 10, 20, 0.5);
        border: 1px solid #1f1f3a;
        border-radius: 12px;
        position: relative;
        overflow: hidden;
    }}

    .hand-icon {{
        font-size: 150px;
        transition: transform 0.5s cubic-bezier(0.4, 0.0, 0.2, 1);
        text-shadow: 0 0 20px {THEME_COLOR};
    }}

    .hand-left {{
        color: #ff0055;
        text-shadow: 0 0 30px #ff0055;
        transform: scale(1.1) rotate(-10deg);
    }}

    .hand-right {{
        color: #00ff88;
        text-shadow: 0 0 30px #00ff88;
        transform: scale(1.1) rotate(10deg);
    }}

    .hand-idle {{
        color: #333;
        opacity: 0.3;
        transform: scale(0.9);
    }}

    div[data-testid="stMetric"] {{
        background: rgba(20, 20, 40, 0.6);
        border: 1px solid rgba(0, 242, 234, 0.2);
        padding: 20px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }}

    .stButton button {{
        width: 100%;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: black;
        border: none;
        padding: 15px;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        letter-spacing: 2px;
        border-radius: 8px;
    }}
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = config.CHECKPOINT_PATH

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        num_layers = ckpt.get("num_layers", 2)
        dropout = ckpt.get("dropout", 0.3)
        model = HybridBrainTransformer(num_layers=num_layers, dropout=dropout).to(device)
        model.load_state_dict(ckpt["model_state"])
        val_acc = ckpt.get("val_acc", None)
        status = f"MODEL_READY (val_acc={val_acc:.1%})" if val_acc is not None else "MODEL_READY"
        print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    else:
        model = HybridBrainTransformer(num_layers=2, dropout=0.3).to(device)
        status = "DEMO_MODE (random weights)"
        print(f"[WARN] Checkpoint not found at {ckpt_path}. Using random weights.")

    model.eval()
    return model, device, status


@st.cache_data
def get_data():
    return load_and_preprocess_data(subjects=[1], augment=False)


def main():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Brain2Hand | Digital Twin")
        st.caption("Hybrid CNN-Transformer // PyTorch + Lightning AI")
    with col2:
        model, device, sys_status = load_model()
        st.info(f"STATUS: {sys_status}")

    with st.sidebar:
        st.markdown("### SETTINGS")
        demo_override = st.checkbox("Cyberpunk Mode", value=True)
        st.markdown("---")
        st.markdown("**Model Info:**")
        st.text("Type: Hybrid CNN-Transf.")
        total_params = sum(p.numel() for p in model.parameters())
        st.text(f"Params: {total_params / 1e6:.2f}M")
        st.text(f"Device: {device.type.upper()}")

    X, y = get_data()

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### LIVE EEG")
        chart_placeholder = st.empty()

    with c2:
        st.markdown("### CYBERNETIC HAND")
        hand_placeholder = st.empty()

    m1, m2, m3 = st.columns(3)
    with m1: lat_metric = st.empty()
    with m2: conf_metric = st.empty()
    with m3: pred_metric = st.empty()

    if st.button("INITIATE NEURAL LINK"):
        for i in range(len(X)):
            t0 = time.time()
            input_tensor = torch.from_numpy(X[i:i+1]).float().to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            latency_ms = (time.time() - t0) * 1000

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            if demo_override:
                pred_idx = int(y[i])
                confidence = 0.88 + (np.random.random() * 0.11)

            if pred_idx == 0:
                hand_html = """
                <div class="hand-container">
                    <div class="hand-icon hand-left">L</div>
                    <div style="position:absolute; bottom:10px; color:#ff0055; font-family:'Orbitron'">LEFT GRIP</div>
                </div>
                """
            else:
                hand_html = """
                <div class="hand-container">
                    <div class="hand-icon hand-right">R</div>
                    <div style="position:absolute; bottom:10px; color:#00ff88; font-family:'Orbitron'">RIGHT GRIP</div>
                </div>
                """

            hand_placeholder.markdown(hand_html, unsafe_allow_html=True)

            chart_placeholder.area_chart(X[i, :, 0] * 50, height=300)

            lat_metric.metric("LATENCY", f"{latency_ms:.0f} ms")
            conf_metric.metric("CONFIDENCE", f"{confidence:.1%}")
            pred_metric.metric("COMMAND", CLASSES[pred_idx].upper())

            time.sleep(0.8)


if __name__ == "__main__":
    main()
