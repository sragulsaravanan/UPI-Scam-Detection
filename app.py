import os
import re
import json
from pathlib import Path

import joblib
import streamlit as st
from scipy.sparse import hstack

try:
    from google import genai
except Exception:
    genai = None


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="UPI Scam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PATHS / CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

with open(BASE_DIR / "model_config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

FINAL_THRESHOLD = float(CONFIG.get("binary_threshold", 0.11))


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px;
}

/* Prevent Streamlit header/toolbar from covering the title */
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.25;
    margin-top: 1rem;
    margin-bottom: 0.45rem;
    padding-top: 0.25rem;
    overflow: visible;
    word-break: normal;
}

.subtitle {
    color: #64748b;
    font-size: 1.02rem;
    margin-bottom: 1rem;
}

.section-card {
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    background: #ffffff;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
}

.result-scam {
    border: 1px solid #fecaca;
    border-radius: 16px;
    padding: 1rem;
    background: #fff7f7;
}

.result-genuine {
    border: 1px solid #bbf7d0;
    border-radius: 16px;
    padding: 1rem;
    background: #f7fff9;
}

.small-note {
    color: #64748b;
    font-size: 0.88rem;
}

.evidence-chip {
    display: inline-block;
    padding: 0.3rem 0.55rem;
    margin: 0.18rem;
    border-radius: 999px;
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    font-size: 0.84rem;
}

/* Extra protection for narrow browser windows */
@media (max-width: 900px) {
    .main-title {
        font-size: 1.8rem;
        line-height: 1.25;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_models():

    word_vec = joblib.load(
        MODEL_DIR / "qr_word_tfidf.pkl"
    )

    char_vec = joblib.load(
        MODEL_DIR / "qr_char_tfidf.pkl"
    )

    binary_svm = joblib.load(
        MODEL_DIR / "qr_augmented_svm.pkl"
    )

    type_vec = joblib.load(
        MODEL_DIR / "scam_type_tfidf.pkl"
    )

    type_svm = joblib.load(
        MODEL_DIR / "scam_type_svm.pkl"
    )

    urgency_vec = joblib.load(
        MODEL_DIR / "urgency_tfidf.pkl"
    )

    urgency_svm = joblib.load(
        MODEL_DIR / "urgency_svm.pkl"
    )

    return (
        word_vec,
        char_vec,
        binary_svm,
        type_vec,
        type_svm,
        urgency_vec,
        urgency_svm,
    )


(
    word_vec,
    char_vec,
    binary_svm,
    type_vec,
    type_svm,
    urgency_vec,
    urgency_svm,
) = load_models()


# =========================================================
# HELPERS
# =========================================================

def normalize_text(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text or "")
    ).strip()


def get_secret(name: str):

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name)


def binary_predict(message: str):

    message = normalize_text(message)

    xw = word_vec.transform(
        [message]
    )

    xc = char_vec.transform(
        [message]
    )

    x = hstack(
        [xw, xc],
        format="csr"
    )

    score = float(
        binary_svm.decision_function(x)[0]
    )

    label = (
        "SCAM"
        if score >= FINAL_THRESHOLD
        else "GENUINE"
    )

    return label, score


def broad_type(message: str):

    pred = type_svm.predict(
        type_vec.transform(
            [normalize_text(message)]
        )
    )[0]

    return pred


def urgency_predict(message: str):

    pred = urgency_svm.predict(
        urgency_vec.transform(
            [normalize_text(message)]
        )
    )[0]

    return pred


def extract_evidence(message: str):

    t = normalize_text(
        message
    ).lower()

    evidence = []

    kyc_terms = [
        "kyc",
        "know your customer",
        "verification",
        "verify your identity",
    ]

    threat_terms = [
        "account will be blocked",
        "account will be frozen",
        "wallet will be frozen",
        "suspended",
        "deactivated",
        "expire",
        "expired",
        "avoid disconnection",
        "service interruption",
    ]

    urgency_terms = [
        "urgent",
        "immediately",
        "now",
        "today",
        "within minutes",
        "within 1 hour",
        "last chance",
        "hurry",
        "act now",
        "deadline",
        "tonight",
    ]

    sensitive_terms = [
        "share your otp",
        "share otp",
        "send otp",
        "enter your otp",
        "otp and pin",
        "upi pin",
        "share your pin",
        "password",
        "cvv",
    ]

    payment_terms = [
        "pay",
        "payment",
        "transfer",
        "debit",
        "credited",
        "transaction",
        "upi",
        "send money",
        "collect request",
    ]

    qr_terms = [
        "qr code",
        "scan qr",
        "scan this qr",
        "qr",
    ]

    reward_terms = [
        "cashback",
        "cash back",
        "reward",
        "prize",
        "lucky draw",
        "won",
        "bonus",
    ]

    if any(
        x in t
        for x in kyc_terms
    ):
        evidence.append(
            "KYC / verification reference"
        )

    if any(
        x in t
        for x in threat_terms
    ):
        evidence.append(
            "Account or wallet threat"
        )

    if any(
        x in t
        for x in urgency_terms
    ):
        evidence.append(
            "Urgent language"
        )

    if any(
        x in t
        for x in sensitive_terms
    ):
        evidence.append(
            "OTP / PIN / sensitive credential request"
        )

    if any(
        x in t
        for x in payment_terms
    ):
        evidence.append(
            "Payment / transaction action"
        )

    if any(
        x in t
        for x in qr_terms
    ):
        evidence.append(
            "QR-code reference"
        )

    if re.search(
        r"https?://|www\.|bit\.ly|tinyurl|t\.co/",
        t
    ):
        evidence.append(
            "External link"
        )

    if any(
        x in t
        for x in reward_terms
    ):
        evidence.append(
            "Reward / cashback language"
        )

    return evidence


def local_explanation(
    label,
    evidence,
    message
):

    if label == "SCAM":

        if evidence:

            clue = ", ".join(
                evidence[:3]
            )

            return (
                "The message was flagged as SCAM by the "
                "machine-learning detector, with supporting "
                f"indicators such as {clue}. These indicators "
                "are shown as evidence for the prediction and "
                "are not independently treated as proof by the "
                "explanation module."
            )

        return (
            "The message was flagged as SCAM by the "
            "machine-learning detector. No single keyword "
            "is treated as proof; the result comes from the "
            "trained text-classification model."
        )

    return (
        "The message was classified as GENUINE by the "
        "machine-learning detector based on its learned "
        "text patterns. A genuine prediction does not "
        "guarantee safety, so users should still verify "
        "unexpected financial requests independently."
    )


def gemini_explanation(
    label,
    evidence,
    message
):

    api_key = get_secret(
        "GEMINI_API_KEY"
    )

    if not api_key or genai is None:

        return (
            local_explanation(
                label,
                evidence,
                message
            ),
            False,
        )

    evidence_text = (
        "; ".join(evidence)
        if evidence
        else "No explicit evidence indicators were detected."
    )

    prompt = f"""
You are the explanation module for an Indian UPI scam SMS detector.

Machine-learning result: {label}

Detected evidence:
{evidence_text}

SMS:
{message}

Write exactly 2 natural sentences.

Use only the SMS and the detected evidence above.
Do not independently reclassify the message.
Do not add facts from outside the SMS.
Do not invent bank/company policies.
Do not mention model internals.
Do not use headings or labels.
"""

    try:

        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = (
            response.text or ""
        ).strip()

        if text:
            return text, True

    except Exception:
        pass

    return (
        local_explanation(
            label,
            evidence,
            message
        ),
        False,
    )


# =========================================================
# DEMO MESSAGES
# =========================================================

DEMO_MESSAGES = {

    "SCAM — KYC + Link":
        "Your bank KYC is expiring today. "
        "Click the link and enter your Aadhaar "
        "and OTP to keep your account active.",

    "SCAM — QR Payment":
        "Congratulations! You won Rs 2,000 cashback. "
        "Scan this QR code and pay Rs 10 to receive "
        "your reward immediately.",

    "SCAM — OTP Request":
        "Your UPI transaction is blocked. "
        "Share the OTP received on your phone now "
        "to cancel the transaction.",

    "GENUINE — Bank OTP":
        "Your ICICI Bank OTP is <OTP>. "
        "OTPs are SECRET. DO NOT disclose it to anyone.",
}


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "### 🛡️ UPI Scam Detector"
    )

    st.markdown(
        "Machine-learning analysis for Indian UPI, "
        "banking, payment and financial SMS."
    )

    st.divider()

    st.markdown(
        "**Quick demo**"
    )

    demo_name = st.selectbox(
        "Choose a sample",
        list(
            DEMO_MESSAGES.keys()
        )
    )

    if st.button(
        "Load selected demo",
        use_container_width=True
    ):

        st.session_state[
            "message_box"
        ] = DEMO_MESSAGES[
            demo_name
        ]

    st.divider()

    st.markdown(
        "**Pipeline**"
    )

    st.write(
        "1. Text preprocessing"
    )

    st.write(
        "2. Word + character TF-IDF"
    )

    st.write(
        "3. Linear SVM scam detection"
    )

    st.write(
        "4. Scam type + urgency"
    )

    st.write(
        "5. Evidence extraction"
    )

    st.write(
        "6. Gemini explanation"
    )


# =========================================================
# MAIN TITLE
# =========================================================

st.markdown(
    """
    <div class="main-title">
        AI-Powered Detection of Indian UPI Scam Messages
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Machine Learning + Evidence Extraction + Explainable AI
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SMS INPUT
# =========================================================

with st.container():

    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        "### Analyze an SMS"
    )

    message = st.text_area(
        "Paste the SMS message below",
        key="message_box",
        height=180,
        placeholder=(
            "Example: Your UPI account will be blocked. "
            "Verify your KYC using the link..."
        ),
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        analyze = st.button(
            "🔍 Analyze Message",
            type="primary",
            use_container_width=True,
        )

    with col2:

        clear = st.button(
            "🧹 Clear",
            use_container_width=True,
        )

    if clear:

        st.session_state[
            "message_box"
        ] = ""

        st.rerun()

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if not normalize_text(message):

        st.warning(
            "Please enter an SMS message first."
        )

        st.stop()

    label, score = binary_predict(
        message
    )

    evidence = extract_evidence(
        message
    )

    risk = urgency_predict(
        message
    )

    scam_type = (
        "Not applicable — message classified as GENUINE"
    )

    if label == "SCAM":

        scam_type = broad_type(
            message
        )

    explanation, used_gemini = (
        gemini_explanation(
            label,
            evidence,
            message
        )
    )

    st.markdown("")

    if label == "SCAM":

        st.markdown(
            """
            <div class="result-scam">
                <h2>🚨 SCAM DETECTED</h2>
                <p>
                    The message crossed the calibrated
                    detection threshold.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="result-genuine">
                <h2>✅ GENUINE</h2>
                <p>
                    The message was not flagged as a scam
                    by the detector.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:

        st.metric(
            "Prediction",
            label
        )

    with c2:

        st.metric(
            "Scam Type",
            scam_type
        )

    with c3:

        st.metric(
            "Urgency",
            risk
        )

    st.markdown("")

    left, right = st.columns(
        2
    )

    with left:

        st.markdown(
            "### 🔎 Detected Evidence"
        )

        if evidence:

            for item in evidence:

                st.markdown(
                    f"""
                    <span class="evidence-chip">
                        {item}
                    </span>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.info(
                "No explicit evidence indicators detected."
            )

        with st.expander(
            "Model details"
        ):

            st.write(
                f"Decision score: `{score:.4f}`"
            )

            st.write(
                f"Calibrated threshold: "
                f"`{FINAL_THRESHOLD:.2f}`"
            )

            st.write(
                "Binary model: Word + Character TF-IDF + Linear SVM"
            )

            st.write(
                "Scam-type model: TF-IDF + Linear SVM"
            )

            st.write(
                "Urgency model: TF-IDF + Linear SVM"
            )

    with right:

        st.markdown(
            "### 💡 Explanation"
        )

        st.info(
            explanation
        )

        if used_gemini:

            st.caption(
                "Explanation generated by Gemini from "
                "the detected evidence."
            )

        else:

            st.caption(
                "Local evidence-grounded fallback explanation "
                "used. Add GEMINI_API_KEY to enable Gemini."
            )

    st.markdown("")

    st.markdown(
        "### 🛡️ Safety advice"
    )

    st.write(
        "Do not share OTPs, UPI PINs, passwords or other "
        "sensitive credentials because an SMS asks you to "
        "do so. Verify unexpected financial requests "
        "through the official banking or payment application."
    )


# =========================================================
# LANDING PAGE
# =========================================================

else:

    st.markdown("")

    a, b, c = st.columns(
        3
    )

    with a:

        st.markdown(
            "### 1. Detect"
        )

        st.write(
            "Classify the SMS as SCAM or GENUINE "
            "using the trained ML model."
        )

    with b:

        st.markdown(
            "### 2. Understand"
        )

        st.write(
            "Identify scam type, urgency level "
            "and supporting evidence."
        )

    with c:

        st.markdown(
            "### 3. Explain"
        )

        st.write(
            "Provide a human-readable explanation "
            "without allowing the LLM to override "
            "the ML result."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Research prototype for Indian UPI, banking, payment "
    "and financial SMS analysis. Predictions are model "
    "outputs and should be treated as decision support, "
    "not as a guarantee of safety."
)
