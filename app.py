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
import pyaudio
import wave

# Set up Streamlit page configuration
st.set_page_config(page_title="🎙️ AI Noise Reducer", layout="centered")

# Title and Subheader
st.title("🎧 AI Noise Reducer")
st.subheader("Denoise your recordings with AI!")

# Choose input method: Upload or Record
option = st.radio("Choose input method:", ["Upload Audio", "Record Live Audio"])

# Model selection (e.g., custom models, demucs, etc.)
model_choice = st.selectbox("Choose model to apply for denoising:", ["Demucs", "Custom Denoiser"])

# Processing status display
status_placeholder = st.empty()

def show_processing_status():
    """Function to show processing feedback."""
    with status_placeholder:
        st.info("🔄 Processing... Please wait while the denoising is happening.")

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

def record_audio(input_path, duration=5, fs=44100):
    """Record live audio and save it as a WAV file using PyAudio."""
    p = pyaudio.PyAudio()

    # Open the audio stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=fs,
                    input=True,
                    frames_per_buffer=1024)
    
    frames = []

    st.write(f"Recording for {duration} seconds...")

    # Record the audio
    for _ in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a file
    with wave.open(input_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

    st.write("Recording finished.")

# Handle the "Upload Audio" option
if option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload your noisy audio (WAV)", type=["wav"])
    
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

            try:
                # Run chosen model (either Demucs or Custom Denoiser)
                if model_choice == "Demucs":
                    demucs_out = run_demucs(input_path)
                    if demucs_out:
                        st.audio(demucs_out, format='audio/wav')
                        st.subheader("Denoised Audio Waveform (Demucs)")
                        plot_waveform(demucs_out, title="Denoised Audio (Demucs)")
                        st.success("✅ Denoising completed using Demucs!")
                    else:
                        st.error("❌ Demucs denoising failed!")

                elif model_choice == "Custom Denoiser":
                    custom_out, img_path = run_custom_denoiser(input_path)
                    if custom_out:
                        st.audio(custom_out, format='audio/wav')
                        st.subheader("Denoised Audio Waveform (Custom Denoiser)")
                        plot_waveform(custom_out, title="Denoised Audio (Custom Denoiser)")
                        st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)
                        st.success("✅ Denoising completed using Custom Denoiser!")
                    else:
                        st.error("❌ Custom Denoiser failed!")

            except Exception as e:
                st.error(f"❌ An error occurred during processing: {e}")

            hide_processing_status()  # Hide processing status
            time.sleep(1)  # Small delay to allow updates to the UI

# Handle the "Record Live Audio" option
elif option == "Record Live Audio":
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
        
        try:
            # Run chosen model (either Demucs or Custom Denoiser)
            if model_choice == "Demucs":
                demucs_out = run_demucs(input_path)
                if demucs_out:
                    st.audio(demucs_out, format='audio/wav')
                    st.subheader("Denoised Audio Waveform (Demucs)")
                    plot_waveform(demucs_out, title="Denoised Audio (Demucs)")
                    st.success("✅ Denoising completed using Demucs!")
                else:
                    st.error("❌ Demucs denoising failed!")

            elif model_choice == "Custom Denoiser":
                custom_out, img_path = run_custom_denoiser(input_path)
                if custom_out:
                    st.audio(custom_out, format='audio/wav')
                    st.subheader("Denoised Audio Waveform (Custom Denoiser)")
                    plot_waveform(custom_out, title="Denoised Audio (Custom Denoiser)")
                    st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)
                    st.success("✅ Denoising completed using Custom Denoiser!")
                else:
                    st.error("❌ Custom Denoiser failed!")

        except Exception as e:
            st.error(f"❌ An error occurred during processing: {e}")

        hide_processing_status()  # Hide processing status
        time.sleep(1)  # Small delay to allow updates to the UI

# ===== Feedback Section =====
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

    # Save feedback
    if not os.path.exists(feedback_file):
        pd.DataFrame([feedback]).to_csv(feedback_file, index=False)
    else:
        pd.DataFrame([feedback]).to_csv(feedback_file, mode='a', header=False, index=False)

    st.success("✅ Thank you for your feedback!")

# Optional: View feedback (for local testing/admin)
if st.checkbox("📊 View Submitted Feedback (for testing only)"):

    if os.path.exists("feedback.csv"):
        st.dataframe(pd.read_csv("feedback.csv"))
    else:
        st.write("No feedback submitted yet.")

st.markdown("---")
st.markdown("Made with ❤️ for sound clarity | [GitHub Repo](https://github.com/yourusername/ai-noise-reducer)")
