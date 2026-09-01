# ============================================================
# HAM10000 SKIN LESION CLASSIFICATION
# Streamlit Deployment Application
# Model: YOLO26M-CLS
# ============================================================

from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from textwrap import dedent
import altair as alt
import torch


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DermaVision AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "final_model"
    / "YOLO26M_HAM10000_FINAL_best.pt"
)


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

# Use GPU (device index 0) when CUDA is available.
# Fall back to CPU for Streamlit Community Cloud and other
# CPU-only environments. This guard makes inference safe on
# both local GPU machines and cloud CPU deployments.
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_INFO = {

    "akiec": {
        "name": "Actinic Keratoses",
        "description":
            "Actinic keratoses and intraepithelial carcinoma "
            "(Bowen's disease)."
    },

    "bcc": {
        "name": "Basal Cell Carcinoma",
        "description":
            "A common form of skin cancer originating from "
            "basal cells."
    },

    "bkl": {
        "name": "Benign Keratosis",
        "description":
            "Benign keratosis-like lesions including "
            "seborrheic keratoses."
    },

    "df": {
        "name": "Dermatofibroma",
        "description":
            "A usually benign fibrous skin lesion."
    },

    "mel": {
        "name": "Melanoma",
        "description":
            "A malignant melanocytic lesion requiring "
            "professional clinical evaluation."
    },

    "nv": {
        "name": "Melanocytic Nevus",
        "description":
            "A melanocytic nevus, commonly known as a mole."
    },

    "vasc": {
        "name": "Vascular Lesion",
        "description":
            "Vascular skin lesions including angiomas and "
            "related vascular abnormalities."
    }
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       Main application
    -------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(38, 99, 235, 0.08),
                transparent 32%
            ),
            linear-gradient(
                180deg,
                #f8fbff 0%,
                #ffffff 45%,
                #f7f9fc 100%
            );
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* --------------------------------------------------------
       Hero section
    -------------------------------------------------------- */

    .hero-container {
        background:
            linear-gradient(
                135deg,
                #07152f 0%,
                #0f2d5c 55%,
                #164e8a 100%
            );

        border-radius: 24px;
        padding: 42px 46px;
        margin-bottom: 26px;

        box-shadow:
            0 20px 50px rgba(15, 45, 92, 0.15);
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;

        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.20);

        color: #dbeafe;
        font-size: 0.80rem;
        font-weight: 600;

        margin-bottom: 18px;
    }

    .hero-title {
        color: #ffffff;
        font-size: 2.65rem;
        font-weight: 750;
        line-height: 1.1;

        margin: 0;
    }

    .hero-subtitle {
        color: #c9d8ee;
        font-size: 1.04rem;
        line-height: 1.7;

        max-width: 760px;
        margin-top: 15px;
        margin-bottom: 0;
    }


    /* --------------------------------------------------------
       Cards
    -------------------------------------------------------- */

    .info-card {
        background: rgba(255,255,255,0.94);

        border: 1px solid #e5eaf2;
        border-radius: 18px;

        padding: 22px;

        box-shadow:
            0 8px 28px rgba(22, 34, 51, 0.06);

        height: 100%;
    }

    .card-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;

        text-transform: uppercase;
        letter-spacing: 0.08em;

        margin-bottom: 8px;
    }

    .card-value {
        color: #0f172a;
        font-size: 1.35rem;
        font-weight: 750;

        margin-bottom: 4px;
    }

    .card-small {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.5;
    }


    /* --------------------------------------------------------
       Prediction result card
    -------------------------------------------------------- */

    .prediction-card {
        background:
            linear-gradient(
                135deg,
                #eff6ff,
                #ffffff
            );

        border: 1px solid #bfdbfe;
        border-radius: 20px;

        padding: 26px;

        box-shadow:
            0 12px 34px rgba(37, 99, 235, 0.08);
    }

    .prediction-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;

        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .prediction-name {
        color: #0f2d5c;
        font-size: 2rem;
        font-weight: 800;

        margin-top: 5px;
        margin-bottom: 5px;
    }

    .prediction-code {
        display: inline-block;

        background: #dbeafe;
        color: #1d4ed8;

        padding: 5px 10px;
        border-radius: 8px;

        font-size: 0.8rem;
        font-weight: 700;
    }


    /* --------------------------------------------------------
       Disclaimer
    -------------------------------------------------------- */

    .disclaimer {
        background: #fff8e6;
        border: 1px solid #f3d48a;
        border-radius: 14px;

        color: #664d03;

        padding: 16px 18px;

        font-size: 0.9rem;
        line-height: 1.55;
    }


    /* --------------------------------------------------------
       Footer
    -------------------------------------------------------- */

    .footer {
        text-align: center;

        color: #94a3b8;
        font-size: 0.82rem;

        margin-top: 45px;
        padding-top: 22px;

        border-top: 1px solid #e5e7eb;
    }


    /* --------------------------------------------------------
       Streamlit uploader
    -------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        background: white;

        border: 1px dashed #93b4df;
        border-radius: 16px;

        padding: 10px;
    }


    /* --------------------------------------------------------
       Button
    -------------------------------------------------------- */

    .stButton > button {
        width: 100%;

        border-radius: 12px;
        border: none;

        padding: 0.75rem 1rem;

        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


# ============================================================
# HELPER FUNCTION
# ============================================================

def predict_image(model, image):

    results = model.predict(
        source=image,
        imgsz=224,
        verbose=False,
        device=DEVICE
    )

    result = results[0]

    probabilities = (
        result.probs.data
        .detach()
        .cpu()
        .numpy()
    )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_code = model.names[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    probability_table = pd.DataFrame({

        "Class Code": [
            model.names[i]
            for i in range(len(probabilities))
        ],

        "Condition": [
            CLASS_INFO[
                model.names[i]
            ]["name"]
            for i in range(len(probabilities))
        ],

        "Probability": probabilities

    })

    probability_table = (
        probability_table
        .sort_values(
            "Probability",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return (
        predicted_code,
        confidence,
        probability_table
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## DermaVision AI")

    st.caption(
        "YOLO26M-based skin lesion classification "
        "research prototype."
    )

    st.divider()

    st.markdown("### Model")

    st.write("**Architecture:** YOLO26M-CLS")
    st.write("**Input Size:** 224 × 224")
    st.write("**Classes:** 7")

    st.divider()

    st.markdown("### Test Performance")

    st.metric(
        "Accuracy",
        "83.16%"
    )

    st.metric(
        "Macro F1",
        "71.77%"
    )

    st.metric(
        "Macro ROC-AUC",
        "96.33%"
    )

    st.divider()

    st.caption(
        "Model trained on HAM10000 using a "
        "lesion-aware split and moderate class oversampling."
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
<div class="hero-container">
<div class="hero-badge">
DEEP LEARNING &bull; DERMOSCOPIC IMAGE ANALYSIS
</div>
<h1 class="hero-title">DermaVision AI</h1>
<p class="hero-subtitle">
An AI-assisted research system for classifying
dermoscopic skin-lesion images across seven
HAM10000 diagnostic categories using a
YOLO26M classification model.
</p>
</div>
""",
    unsafe_allow_html=True
)

# ============================================================
# MODEL INFORMATION CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        dedent("""
        <div class="info-card">
            <div class="card-label">Architecture</div>
            <div class="card-value">YOLO26M-CLS</div>
            <div class="card-small">
                Image classification model
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        dedent("""
        <div class="info-card">
            <div class="card-label">Dataset</div>
            <div class="card-value">HAM10000</div>
            <div class="card-small">
                7 lesion categories
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        dedent("""
        <div class="info-card">
            <div class="card-label">Test Accuracy</div>
            <div class="card-value">83.16%</div>
            <div class="card-small">
                1,249 / 1,502 correct
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        dedent("""
        <div class="info-card">
            <div class="card-label">Macro ROC-AUC</div>
            <div class="card-value">96.33%</div>
            <div class="card-small">
                Multiclass one-vs-rest
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    dedent("""
    <div class="disclaimer">
        <strong>Research-use notice:</strong>
        This application is an academic AI prototype and is not
        a medical diagnostic device. Predictions must not replace
        examination, dermoscopy assessment, histopathology, or
        advice from a qualified healthcare professional.
    </div>
    """),
    unsafe_allow_html=True
)


st.write("")
st.write("")


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as error:

    st.error(
        "Unable to load the trained model."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown("## Skin Lesion Assessment")

st.caption(
    "Upload one dermoscopic lesion image in JPG, JPEG, or PNG format."
)

uploaded_file = st.file_uploader(
    "Upload lesion image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        image_bytes = uploaded_file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        image_column, result_column = st.columns(
            [1.05, 1.35],
            gap="large"
        )


        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        with image_column:

            st.markdown("### Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

            st.caption(
                f"Image size: "
                f"{image.width} × {image.height} pixels"
            )


        # ----------------------------------------------------
        # Run prediction
        # ----------------------------------------------------

        with st.spinner(
            "Analyzing dermoscopic image..."
        ):

            (
                predicted_code,
                confidence,
                probability_table
            ) = predict_image(
                model,
                image
            )


        predicted_info = CLASS_INFO[
            predicted_code
        ]


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        with result_column:

            st.markdown("### AI Prediction")

            st.markdown(
                f"""
<div class="prediction-card">
<div class="prediction-label">Predicted Category</div>
<div class="prediction-name">{predicted_info["name"]}</div>
<div class="prediction-code">{predicted_code.upper()}</div>
<br><br>
<div class="card-label">Model Confidence</div>
<div class="card-value">{confidence * 100:.2f}%</div>
<div class="card-small">{predicted_info["description"]}</div>
</div>
""",
                unsafe_allow_html=True
            )

            st.write("")

            st.progress(
                min(
                    int(confidence * 100),
                    100
                )
            )


        # ====================================================
        # PROBABILITY DISTRIBUTION
        # ====================================================

        st.write("")
        st.divider()

        st.markdown(
            "## Prediction Probability Distribution"
        )

        st.caption(
            "Model confidence across all seven diagnostic categories."
        )

        probability_display = (
            probability_table[
                [
                    "Condition",
                    "Probability"
                ]
            ]
            .copy()
        )

        probability_display[
            "Probability"
        ] = (
            probability_display[
                "Probability"
            ] * 100
        )

        # Create Altair horizontal bar chart
        chart = alt.Chart(probability_display).mark_bar().encode(
            x=alt.X(
                "Probability:Q",
                scale=alt.Scale(domain=[0, 100]),
                title="Probability (%)"
            ),
            y=alt.Y(
                "Condition:N",
                sort="-x",
                title="Condition"
            ),
            tooltip=["Condition:N", alt.Tooltip("Probability:Q", format=".2f")]
        ).properties(
            height=250,
            width=600
        ).interactive()

        st.altair_chart(chart, use_container_width=True)


        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        st.markdown("### Top 3 Predictions")

        top_three = probability_table.head(3)

        columns = st.columns(3)

        for column, (_, row) in zip(
            columns,
            top_three.iterrows()
        ):

            with column:

                prob_percentage = row['Probability'] * 100
                
                if prob_percentage < 0.01:
                    value_display = "<0.01%"
                else:
                    value_display = f"{prob_percentage:.2f}%"

                st.metric(
                    label=row["Condition"],
                    value=value_display
                )


        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

        with st.expander(
            "View detailed probability table"
        ):

            detailed_table = (
                probability_table.copy()
            )

            detailed_table[
                "Probability"
            ] = (
                detailed_table[
                    "Probability"
                ] * 100
            ).round(2)

            detailed_table.rename(
                columns={
                    "Probability":
                        "Probability (%)"
                },
                inplace=True
            )

            st.dataframe(
                detailed_table,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # CLINICAL CAUTION
        # ====================================================

        if predicted_code == "mel":

            st.warning(
                "The model's highest-scoring category is melanoma. "
                "This AI result is not a diagnosis. Professional "
                "dermatological evaluation is required for any "
                "suspicious lesion."
            )

        else:

            st.info(
                "AI classifications can be incorrect, including "
                "confusion between melanoma and benign lesions. "
                "Clinical assessment should take priority over "
                "the model output."
            )


    except Exception as error:

        st.error(
            "The uploaded image could not be processed."
        )

        st.code(str(error))


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown(
        """
<div style="text-align:center;padding:55px 20px;border:1px dashed #cbd5e1;border-radius:18px;background:rgba(255,255,255,0.70);">
<div style="font-size:2.5rem;margin-bottom:12px;">🔬</div>
<div style="color:#0f172a;font-size:1.15rem;font-weight:700;">Upload a dermoscopic image to begin</div>
<div style="color:#64748b;margin-top:6px;">Supported formats: JPG, JPEG and PNG</div>
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# MODEL DETAILS
# ============================================================

st.write("")
st.write("")

with st.expander(
    "About the research model"
):

    st.markdown(
        """
        **Model:** YOLO26M-CLS  
        **Dataset:** HAM10000  
        **Number of classes:** 7  
        **Input resolution:** 224 × 224  
        **Split strategy:** Lesion-aware train/validation/test split  
        **Imbalance strategy:** Moderate training-set oversampling  
        **Test accuracy:** 83.16%  
        **Balanced accuracy:** 68.98%  
        **Macro F1-score:** 71.77%  
        **Macro ROC-AUC:** 96.33%

        The model demonstrated improved minority-class performance
        after oversampling, although melanoma-to-nevus confusion
        remains an important limitation of the system.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    dedent("""
    <div class="footer">
        DermaVision AI • YOLO26M-CLS • HAM10000 Research Prototype
        <br>
        Built for academic research and educational use.
    </div>
    """),
    unsafe_allow_html=True
)
