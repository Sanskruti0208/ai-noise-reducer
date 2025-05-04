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
import streamlit.components.v1 as components
from language_texts import texts

# Set up Streamlit page configuration
st.set_page_config(page_title="🎙️ AI Noise Reducer", layout="centered")

# Language selection
language = st.selectbox("Choose language for instructions:", ["English", "Spanish", "French", "Marathi", "Hindi"])

# Set language-specific texts
selected_text = texts[language]

# Title and Subheader
st.title(selected_text["title"])
st.subheader(selected_text["subheader"])

# Choose input method: Upload or Record
option = st.radio("Choose input method:", [selected_text["upload_audio"], selected_text["record_audio"]])

# Model selection (e.g., custom models, demucs, etc.)
model_choice = st.selectbox(selected_text["choose_model"], ["Demucs", "Custom Denoiser"])

# Processing status display
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
    uploaded_file = st.file_uploader("Upload your noisy audio (WAV)")
    if uploaded_file:
        input_path = os.path.join("data", "recorded_audio", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(input_path, format='audio/wav')
        st.subheader("Original Audio Waveform")
        plot_waveform(input_path, title="Original Noisy Audio")

        if st.button("Run Denoising"):
            show_processing_status()
            if model_choice == "Demucs":
                demucs_out = run_demucs(input_path)
                hide_processing_status()
                st.success(selected_text["denoising_complete"].format(model="Demucs"))
                if demucs_out:
                    st.audio(demucs_out, format='audio/wav')
                    st.subheader("Denoised Audio Waveform (Demucs)")
                    plot_waveform(demucs_out, title="Denoised Audio (Demucs)")
            elif model_choice == "Custom Denoiser":
                custom_out, img_path = run_custom_denoiser(input_path)
                hide_processing_status()
                st.success(selected_text["denoising_complete"].format(model="Custom Denoiser"))
                if custom_out:
                    st.audio(custom_out, format='audio/wav')
                    st.subheader("Denoised Audio Waveform (Custom Denoiser)")
                    plot_waveform(custom_out, title="Denoised Audio (Custom Denoiser)")
                    st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)
            time.sleep(1)

# Record Audio (Client-Side)
elif option == selected_text["record_audio"]:
    if st.button("🎙️ Start Recording"):
        show_processing_status()
        audio_path = record_audio()
        hide_processing_status()

        if audio_path:
            st.audio(audio_path, format='audio/wav')
            st.subheader("Recorded Audio Waveform")
            plot_waveform(audio_path, title="Recorded Audio")

            if st.button("Run Denoising"):
                if model_choice == "Demucs":
                    output_path = run_demucs(audio_path)
                else:
                    output_path, img_path = run_custom_denoiser(audio_path)

                st.success(selected_text["denoising_complete"].format(model=model_choice))
                st.audio(output_path, format='audio/wav')
                plot_waveform(output_path, title=f"Denoised Audio ({model_choice})")


# Feedback Section
st.markdown("---")
st.subheader("🗣️ Share Your Feedback")
st.markdown("How would you rate the audio quality after denoising?")
rating = st.radio("Overall audio quality:", ["Excellent", "Good", "Average", "Poor"], key="rating")
comment = st.text_area("💬 Additional Comments", placeholder="Any suggestions or issues you noticed...")

if st.button("Submit Feedback"):
    feedback = {
        "timestamp": datetime.datetime.now().isoformat(),
        "rating": rating,
        "comment": comment
    }
    feedback_file = "feedback.csv"
    if not os.path.exists(feedback_file):
        pd.DataFrame([feedback]).to_csv(feedback_file, index=False)
    else:
        pd.DataFrame([feedback]).to_csv(feedback_file, mode='a', header=False, index=False)
    st.success("✅ Thank you for your feedback!")

if st.checkbox("📊 View Submitted Feedback (for testing only)"):
    if os.path.exists("feedback.csv"):
        st.dataframe(pd.read_csv("feedback.csv"))
    else:
        st.write("No feedback submitted yet.")

st.markdown("---")
st.markdown("Made with ❤️ for sound clarity | [GitHub Repo](https://github.com/yourusername/ai-noise-reducer)")
