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
import json

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
st.info(texts[language].get("tip", ""))
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
    uploaded_file = st.file_uploader(texts[lang]["upload_noisy_audio"], type=["wav"])
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

# Record Audio using audiorecorder
elif option == selected_text["record_audio"]:
    st.subheader("🎙️ Record Audio Below")
    audio = audiorecorder("Click to record", "Recording...")

    if len(audio) > 0:
        save_dir = "data/recorded_audio"
        os.makedirs(save_dir, exist_ok=True)
        recorded_wav_path = os.path.join(save_dir, "recorded.wav")

        audio.export(recorded_wav_path, format="wav")

        st.success("✅ Audio recorded and saved!")
        st.audio(recorded_wav_path, format='audio/wav')
        plot_waveform(recorded_wav_path, title="Recorded Audio")

        if st.button("Run Denoising"):
            show_processing_status()
            if model_choice == "Demucs":
                output_path = run_demucs(recorded_wav_path)
            else:
                output_path, img_path = run_custom_denoiser(recorded_wav_path)
            hide_processing_status()

            st.success(selected_text["denoising_complete"].format(model=model_choice))
            st.audio(output_path, format='audio/wav')
            plot_waveform(output_path, title=f"Denoised Audio ({model_choice})")
            if model_choice == "Custom Denoiser":
                st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)

# feedback section
st.subheader(selected_text["feedback_title"])

with st.form("feedback_form"):
    user_feedback = st.text_area(selected_text["feedback_placeholder"])
    star_rating = st.slider(selected_text["rating_prompt"], 1, 5, 5)

    # Rating labels
    rating_labels = {
        1: "😞 Poor",
        2: "😐 Fair",
        3: "🙂 Good",
        4: "😀 Very Good",
        5: "🌟 Excellent"
    }
    st.markdown(f"**Selected Rating:** {rating_labels[star_rating]}")

    submit_feedback = st.form_submit_button("Submit Feedback")

if submit_feedback:
    feedback_data = {
        "timestamp": str(datetime.datetime.now()),
        "feedback": user_feedback,
        "rating": star_rating,
        "label": rating_labels[star_rating]
    }

    # Save feedback to JSON file
    if not os.path.exists("feedbacks.json"):
        with open("feedbacks.json", "w") as f:
            json.dump([], f, indent=4)  # Create an empty list if the file doesn't exist

    try:
        with open("feedbacks.json", "r") as f:
            existing_data = json.load(f)
    except json.JSONDecodeError:
        existing_data = []

    existing_data.append(feedback_data)

    with open("feedbacks.json", "w") as f:
        json.dump(existing_data, f, indent=4)

    st.success(selected_text["thank_you_feedback"])

# Display Previous Feedbacks
if os.path.exists("feedbacks.json"):
    with open("feedbacks.json", "r") as f:
        all_feedbacks = json.load(f)

    if all_feedbacks:
        st.subheader(selected_text["previous_feedbacks"])
        with st.expander(selected_text["see_all_feedbacks"]):
            for item in reversed(all_feedbacks):
                st.markdown(f"- 🗓️ **{item['timestamp']}** | ⭐ {item['rating']} stars ({item['label']})")
                st.markdown(f"  > {item['feedback']}")

st.markdown("---")
st.markdown("Made with ❤️ for sound clarity | [GitHub Repo](https://github.com/yourusername/ai-noise-reducer)")
