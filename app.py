import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import plotly.express as px

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# ---------------- LOAD FILES ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("Resume.csv")

model = load_model()
tokenizer = load_tokenizer()
encoder = load_encoder()
df = load_data()

# ---------------- SIDEBAR ----------------

st.sidebar.title("📄 AI Resume Analyzer")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Resume Prediction",
        "Dataset Analytics"
    ]
)

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.title("📄 AI Resume Classification System")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Resumes",
        len(df)
    )

    col2.metric(
        "Categories",
        df["Category"].nunique()
    )

    col3.metric(
        "Dataset Size",
        f"{df.shape[0]}"
    )

    st.markdown("---")

    st.subheader("Sample Dataset")

    st.dataframe(
        df[["Category"]].head(20),
        use_container_width=True
    )

# ---------------- PREDICTION ----------------

elif page == "Resume Prediction":

    st.title("🤖 Resume Category Prediction")

    resume_text = st.text_area(
        "Paste Resume Text",
        height=300
    )

    if st.button("Predict Category"):

        if resume_text.strip():

            seq = tokenizer.texts_to_sequences(
                [resume_text]
            )

            padded = tf.keras.preprocessing.sequence.pad_sequences(
                seq,
                maxlen=500
            )

            pred = model.predict(padded)

            class_id = np.argmax(pred)

            category = encoder.inverse_transform(
                [class_id]
            )[0]

            confidence = np.max(pred) * 100

            st.success(
                f"Predicted Category: {category}"
            )

            st.info(
                f"Confidence: {confidence:.2f}%"
            )

# ---------------- ANALYTICS ----------------

else:

    st.title("📊 Resume Dataset Analytics")

    category_counts = (
        df["Category"]
        .value_counts()
        .reset_index()
    )

    category_counts.columns = [
        "Category",
        "Count"
    ]

    fig = px.bar(
        category_counts,
        x="Category",
        y="Count",
        title="Resume Categories Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    df["Resume_Length"] = (
        df["Resume_str"]
        .astype(str)
        .apply(lambda x: len(x.split()))
    )

    fig2 = px.histogram(
        df,
        x="Resume_Length",
        nbins=40,
        title="Resume Length Distribution"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )
