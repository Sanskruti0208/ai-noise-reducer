import streamlit as st
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from denoise_audio import run_custom_denoiser, run_demucs, record_audio
import time
import numpy as np
import datetime
import pandas as pd
from language_texts import texts  # Import the language texts

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
    """Function to show processing feedback."""
    with status_placeholder:
        st.info(selected_text["process_status"])

def hide_processing_status():
    """Function to hide processing feedback after completion."""
    with status_placeholder:
        st.empty()

def plot_waveform(audio_path, title="Waveform"):
    """Plot the waveform of the audio."""
    y, sr = librosa.load(audio_path, sr=None)  # Load the audio file
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

# Handle the "Upload Audio" option
if option == selected_text["upload_audio"]:
    uploaded_file = st.file_uploader(f"Upload your noisy audio (WAV)")

    if uploaded_file:
        # Save the uploaded audio file
        input_path = os.path.join("data", "recorded_audio", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        # Show uploaded audio preview
        st.audio(input_path, format='audio/wav')

        # Plot waveform of the original noisy audio
        st.subheader("Original Audio Waveform")
        plot_waveform(input_path, title="Original Noisy Audio")

        # Start Denoising when the button is clicked
        if st.button("Run Denoising"):
            show_processing_status()  # Show processing status

            # Run chosen model (either Demucs or Custom Denoiser)
            if model_choice == "Demucs":
                demucs_out = run_demucs(input_path)
                hide_processing_status()  # Hide processing status
                st.success(selected_text["denoising_complete"].format(model="Demucs"))
                if demucs_out:
                    st.audio(demucs_out, format='audio/wav')

                    # Plot waveform of the denoised audio
                    st.subheader("Denoised Audio Waveform (Demucs)")
                    plot_waveform(demucs_out, title="Denoised Audio (Demucs)")

            elif model_choice == "Custom Denoiser":
                custom_out, img_path = run_custom_denoiser(input_path)
                hide_processing_status()  # Hide processing status
                st.success(selected_text["denoising_complete"].format(model="Custom Denoiser"))
                if custom_out:
                    st.audio(custom_out, format='audio/wav')

                    # Plot waveform of the denoised audio
                    st.subheader("Denoised Audio Waveform (Custom Denoiser)")
                    plot_waveform(custom_out, title="Denoised Audio (Custom Denoiser)")

                    # Display comparison image
                    st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)

            time.sleep(1)  # Small delay to allow updates to the UI

# Handle the "Record Live Audio" option
elif option == selected_text["record_audio"]:
    if st.button("Start Recording"):
        # Record live audio and save
        input_path = os.path.join("data", "recorded_audio", "live_record.wav")
        record_audio(input_path)

        # Show recording audio preview
        st.audio(input_path, format='audio/wav')

        # Plot waveform of the recorded live audio
        st.subheader("Recorded Live Audio Waveform")
        plot_waveform(input_path, title="Recorded Live Audio")

        # Show processing status
        show_processing_status()

        # Run chosen model (either Demucs or Custom Denoiser)
        if model_choice == "Demucs":
            demucs_out = run_demucs(input_path)
            hide_processing_status()  # Hide processing status
            st.success(selected_text["denoising_complete"].format(model="Demucs"))
            if demucs_out:
                st.audio(demucs_out, format='audio/wav')

                # Plot waveform of the denoised audio
                st.subheader("Denoised Audio Waveform (Demucs)")
                plot_waveform(demucs_out, title="Denoised Audio (Demucs)")

        elif model_choice == "Custom Denoiser":
            custom_out, img_path = run_custom_denoiser(input_path)
            hide_processing_status()  # Hide processing status
            st.success(selected_text["denoising_complete"].format(model="Custom Denoiser"))
            if custom_out:
                st.audio(custom_out, format='audio/wav')

                # Plot waveform of the denoised audio
                st.subheader("Denoised Audio Waveform (Custom Denoiser)")
                plot_waveform(custom_out, title="Denoised Audio (Custom Denoiser)")

                # Display comparison image
                st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)

        time.sleep(1)  # Small delay to allow updates to the UI

# ===== Feedback Section =====
st.markdown("---")
st.subheader(selected_text["feedback_section"])

st.markdown(selected_text["rate_quality"])
