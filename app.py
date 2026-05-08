import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
from pipeline import AnimalAnalysisPipeline, CLASS_NAMES, DEVICE
import json

@st.cache_resource
def load_animal_info():
    with open("animal_info.json", "r") as f:
        return json.load(f)

animal_info = load_animal_info()

st.set_page_config(page_title="Animal Image Analysis", page_icon="🦁", layout="wide")

st.sidebar.title("Configuration")
st.sidebar.markdown(f"**Device:** {DEVICE}")
st.sidebar.markdown(f"**Classes:** {len(CLASS_NAMES)}")


st.title("🦁 Animal Image Analysis")
st.markdown("##### AI-powered wildlife detection, classification and biological intelligence")
st.markdown("---")

if "pipeline" not in st.session_state:
    with st.spinner("Loading models..."):
        st.session_state.pipeline = AnimalAnalysisPipeline(
            yolo_weights="yolov8n.pt",
            classifier_weights="weights/efficientnet_b0_best.pth",
            num_classes=len(CLASS_NAMES),
            class_names=CLASS_NAMES,
        )

pipeline = st.session_state.pipeline

uploaded = st.file_uploader("Upload an animal image (JPG / PNG)", type=["jpg","jpeg","png"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("Analyzing..."):
        result = pipeline.run(bgr)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Species", result["predicted_class"].capitalize())
    col2.metric("Confidence", f"{result['confidence']:.1%}")
    col3.metric("Detection (ms)", f"{result['detection_ms']:.1f}")
    col4.metric("Total latency (ms)", f"{result['total_ms']:.1f}")

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.subheader("Original")
    c1.image(result["original_rgb"], use_container_width=True)
    c2.subheader("Detected region")
    c2.image(result["annotated_rgb"], use_container_width=True)
    c3.subheader("Cropped ROI")
    c3.image(result["roi_rgb"], use_container_width=True)
    c4.subheader("Grad-CAM heatmap")
    c4.image(result["heatmap_rgb"], use_container_width=True)
    c4.caption("Warm regions = model focus area")

    st.markdown("---")
    st.subheader("Top 5 Predictions")

    probs = result["probabilities"]
    top5_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]

    for rank, idx in enumerate(top5_idx):
        name = CLASS_NAMES[idx].capitalize()
        prob = float(probs[idx])

        if rank == 0:
            color = "🟢"
            label = f"**{rank+1}. {name}**"
        elif rank == 1:
            color = "🟡"
            label = f"{rank+1}. {name}"
        else:
            color = "🔴"
            label = f"{rank+1}. {name}"

        col_name, col_bar, col_pct = st.columns([2, 5, 1])
        col_name.markdown(f"{color} {label}")
        col_bar.progress(prob)
        col_pct.markdown(f"**{prob:.1%}**")

    st.markdown("---")
    st.subheader("🦁 Animal Intelligence Panel")

    predicted = result["predicted_class"].lower()
    info = animal_info.get(predicted, None)

    if info:
        # Conservation status banner
        status = info["conservation_status"]
        if info["is_endangered"]:
            st.error(f"🔴 ENDANGERED SPECIES — {status}")
        elif status == "Vulnerable":
            st.warning(f"🟡 Vulnerable Species — {status}")
        elif status == "Near Threatened":
            st.warning(f"🟡 Near Threatened — {status}")
        else:
            st.success(f"🟢 {status}")

        # Tabs
        tab_a, tab_b, tab_c, tab_d = st.tabs([
            "📋 Overview",
            "🌍 Conservation",
            "🏕️ Habitat",
            "💡 Facts"
        ])

        with tab_a:
            c1, c2, c3 = st.columns(3)
            c1.metric("Scientific Name", info["scientific_name"])
            c2.metric("Diet", info["diet"])
            c3.metric("Lifespan", info["lifespan"])

        with tab_b:
            c1, c2 = st.columns(2)
            c1.metric("Conservation Status", info["conservation_status"])
            c2.metric("Population", info["population"])
            if info["protected"]:
                st.info(f"🛡️ Protected Species — {info['protection_details']}")
            else:
                st.warning("⚠️ Not legally protected in most regions")

        with tab_c:
            st.markdown(f"**Habitat:** {info['habitat']}")
            st.markdown(f"**Regions:** {', '.join(info['regions'])}")

        with tab_d:
            st.info(f"💡 **Fun Fact:** {info['fun_fact']}")
    else:
        st.warning("No information available for this animal")
    if result["confidence"] < 0.50:
        st.warning("⚠️ Low confidence — result may be incorrect. Try a clearer image.")
    elif result["confidence"] < 0.75:
        st.info("ℹ️ Moderate confidence — consider verifying the result.")

    st.markdown("---")
st.subheader("📊 Evaluation Dashboard")

tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Sample Predictions", "Metrics"])

with tab1:
    st.image("outputs/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Correct Predictions")
        st.image("outputs/sample_predictions/correct_predictions.png", use_container_width=True)
    with col2:
        st.markdown("### ❌ Incorrect Predictions")
        st.image("outputs/sample_predictions/incorrect_predictions.png", use_container_width=True)

with tab3:
    import json
    metrics = json.load(open("outputs/metrics_summary.json"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{metrics['accuracy']:.2%}")
    m2.metric("Precision", f"{metrics['precision']:.2%}")
    m3.metric("Recall",    f"{metrics['recall']:.2%}")
    m4.metric("F1-Score",  f"{metrics['f1_score']:.2%}")
    st.markdown("---")
    st.markdown("### Classification Report")
    report = open("outputs/classification_report.txt").read()
    st.text(report)


st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:13px'>"
    "🦁 Animal Image Analysis System — YOLO + EfficientNet + Grad-CAM | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)