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

# Ensure required folders exist
os.makedirs("data/recorded_audio", exist_ok=True)

# Streamlit page config
st.set_page_config(page_title="🎙️ AI Noise Reducer", layout="centered")
st.title("🎧 AI Noise Reducer")
st.subheader("Denoise your recordings with AI!")

# UI controls
option = st.radio("Choose input method:", ["Upload Audio", "Record Live Audio"])
model_choice = st.selectbox("Choose model to apply for denoising:", ["Demucs", "Custom Denoiser"])
status_placeholder = st.empty()

# === Helper Functions ===

def show_processing_status():
    with status_placeholder:
        st.info("🔄 Processing... Please wait.")

def hide_processing_status():
    with status_placeholder:
        st.empty()

def plot_waveform(audio_path, title="Waveform"):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"⚠️ Error plotting waveform: {e}")

def record_audio(input_path, duration=5, fs=44100):
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=fs,
                        input=True,
                        frames_per_buffer=1024)
        frames = []
        st.write(f"🎙️ Recording for {duration} seconds...")
        for _ in range(0, int(fs / 1024 * duration)):
            frames.append(stream.read(1024))
        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(input_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(fs)
            wf.writeframes(b''.join(frames))

        st.success("🎉 Recording finished!")

    except Exception as e:
        st.error(f"⚠️ Error during recording: {e}")

# === Main Logic ===

def process_audio(input_path):
    show_processing_status()
    try:
        if model_choice == "Demucs":
            output = run_demucs(input_path)
            if output:
                st.audio(output, format='audio/wav')
                st.subheader("📊 Denoised Waveform (Demucs)")
                plot_waveform(output)
                st.success("✅ Denoising complete with Demucs!")
            else:
                st.error("❌ Demucs failed.")

        elif model_choice == "Custom Denoiser":
            output, img_path = run_custom_denoiser(input_path)
            if output:
                st.audio(output, format='audio/wav')
                st.subheader("📊 Denoised Waveform (Custom)")
                plot_waveform(output)
                st.image(img_path, caption="Comparison: Noisy vs Denoised", use_column_width=True)
                st.success("✅ Denoising complete with Custom Denoiser!")
            else:
                st.error("❌ Custom Denoiser failed.")
    except Exception as e:
        st.error(f"❌ Processing error: {e}")
    finally:
        hide_processing_status()
        time.sleep(1)

# === Upload Audio ===

if option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload a WAV file:", type=["wav"])
    if uploaded_file:
        input_path = os.path.join("data", "recorded_audio", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())
        st.audio(input_path, format='audio/wav')
        st.subheader("📈 Original Audio Waveform")
        plot_waveform(input_path)
        if st.button("Run Denoising"):
            process_audio(input_path)

# === Record Live Audio ===

elif option == "Record Live Audio":
    if st.button("Start Recording"):
        input_path = os.path.join("data", "recorded_audio", "live_record.wav")
        record_audio(input_path)
        st.audio(input_path, format='audio/wav')
        st.subheader("📈 Recorded Audio Waveform")
        plot_waveform(input_path)
        process_audio(input_path)

# === Feedback Section ===

st.markdown("---")
st.subheader("🗣️ Share Your Feedback")
rating = st.radio("Rate the denoised audio:", ["Excellent", "Good", "Average", "Poor"])
comment = st.text_area("💬 Additional Comments", placeholder="Your suggestions or issues...")

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

# Optional: Developer feedback viewer (can be removed in production)
if st.checkbox("📊 View Feedback Data (Admin Only)"):
    if os.path.exists("feedback.csv"):
        df = pd.read_csv("feedback.csv")
        st.dataframe(df)
    else:
        st.info("No feedback yet.")
