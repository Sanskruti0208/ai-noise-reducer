import streamlit as st
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from denoise_audio import run_custom_denoiser, run_demucs
import time
import numpy as np
import datetime
import pandas as pd
from language_texts import texts
from scipy.io.wavfile import write
import base64
import io
import streamlit.components.v1 as components
from audiorecorder import audiorecorder
import csv

# Set up Streamlit page configuration
st.set_page_config(page_title="🎙️ AI Noise Reducer", layout="centered")

# Language selection
language = st.selectbox("Choose language for instructions:", ["English","Marathi", "Hindi"])
st.session_state.language = language 

# Get selected text for the current language
selected_text = texts[st.session_state.language]

# Title and Subheader
st.title(selected_text["title"])
st.subheader(selected_text["subheader"])

# Input method selection
option = st.radio("Choose input method:", [selected_text["upload_audio"], selected_text["record_audio"]])
st.info(selected_text.get("tip", ""))
model_choice = st.selectbox(selected_text["choose_model"], ["Custom Denoiser", "Demucs"])
status_placeholder = st.empty()

def show_processing_status():
    with status_placeholder:
        st.info(selected_text["process_status"])

def hide_processing_status():
    with status_placeholder:
        st.empty()

def plot_waveform(audio_path, title="Waveform"):
    y, sr = librosa.load(audio_path, sr=None)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

# Upload Audio
if option == selected_text["upload_audio"]:
    uploaded_file = st.file_uploader(selected_text["upload_noisy_audio"], type=["wav"])
    if uploaded_file:
        input_path = os.path.join("data", "recorded_audio", uploaded_file.name)
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(input_path, format='audio/wav')
        st.subheader("Original Audio Waveform")
        plot_waveform(input_path, title="Original Noisy Audio")

        if st.button("Run Denoising"):
            show_processing_status()
            if model_choice == "Demucs":
                output_path = run_demucs(input_path)
            else:
                output_path, img_path = run_custom_denoiser(input_path)
            hide_processing_status()

            st.success(selected_text["denoising_complete"].format(model=model_choice))
            st.audio(output_path, format='audio/wav')
            plot_waveform(output_path, title=f"Denoised Audio ({model_choice})")
            if model_choice == "Custom Denoiser":
                st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)
                
# Setup file paths
feedback_dir = "data"
feedback_file = os.path.join(feedback_dir, "feedbacks.csv")
os.makedirs(feedback_dir, exist_ok=True)

# Rating labels
rating_labels = {
    1: "😞 Poor",
    2: "😐 Fair",
    3: "🙂 Good",
    4: "😀 Very Good",
    5: "🌟 Excellent"
}

# Feedback section UI
st.subheader("🗣️ Share Your Feedback")

with st.form(key="feedback_form"):
    user_feedback = st.text_area("Write your feedback:")
    star_rating = st.slider("Rate this app (1 to 5 stars)", 1, 5, 5)
    st.markdown(f"**Selected Rating:** {rating_labels[star_rating]}")
    submit_feedback = st.form_submit_button("Submit Feedback")

# Save feedback to CSV
if submit_feedback:
    feedback_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feedback": user_feedback.strip() if user_feedback.strip() else "No comment",
        "rating": star_rating,
        "label": rating_labels[star_rating]
    }

    file_exists = os.path.isfile(feedback_file)
    try:
        with open(feedback_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "feedback", "rating", "label"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(feedback_entry)

        st.success("🙏 Thank you for your feedback!")
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

# Show previous feedbacks
if os.path.exists(feedback_file):
    try:
        df = pd.read_csv(feedback_file)
        if not df.empty:
            st.subheader("📋 Previous Feedbacks")
            with st.expander("👁️ See All Feedbacks"):
                for _, row in df[::-1].iterrows():
                    st.markdown(f"- 🗓️ **{row['timestamp']}** | ⭐ {row['rating']} stars ({row['label']})")
                    st.markdown(f"  > {row['feedback']}")
        else:
            st.info("No feedbacks yet.")
    except Exception as e:
        st.warning(f"⚠️ Unable to read previous feedbacks: {e}")
        
st.markdown("---")
st.markdown("Made with ❤️ for sound clarity | [GitHub Repo](https://github.com/yourusername/ai-noise-reducer)")
