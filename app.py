import streamlit as st
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.axes import Axes  # Explicit import of Axes
from denoise_audio import run_custom_denoiser, run_demucs
import time
import numpy as np
import datetime
import pandas as pd
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import av
import queue
import soundfile as sf

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
        fig, ax = plt.subplots(figsize=(10, 4))  # Create figure and axis
        librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax)  # Plot the waveform
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"⚠️ Error plotting waveform: {e}")

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

# === Record Live Audio with WebRTC ===

class AudioRecorder(AudioProcessorBase):
    def __init__(self) -> None:
        self.audio_buffer = queue.Queue()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten()
        self.audio_buffer.put(audio)
        return frame

if option == "Record Live Audio":
    audio_recorder = AudioRecorder()

    webrtc_ctx = webrtc_streamer(
        key="audio",
        mode="sendonly",
        in_audio_enabled=True,
        client_settings=ClientSettings(
            media_stream_constraints={"video": False, "audio": True}
        ),
        audio_processor_factory=lambda: audio_recorder,
        async_processing=True,
    )

    if st.button("Save Recording"):
        if webrtc_ctx.state.playing:
            st.warning("🎙️ Capturing... please wait 5 seconds.")
            time.sleep(5)

            audio_data = []
            while not audio_recorder.audio_buffer.empty():
                audio_data.extend(audio_recorder.audio_buffer.get())

            input_path = os.path.join("data", "recorded_audio", "live_record.wav")
            sf.write(input_path, np.array(audio_data), samplerate=48000)
            st.success("✅ Recording saved!")
            st.audio(input_path, format='audio/wav')
            st.subheader("📈 Recorded Audio Waveform")
            plot_waveform(input_path)
            process_audio(input_path)
        else:
            st.error("❌ Start the recording first!")

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

# Optional: Developer feedback viewer
if st.checkbox("📊 View Feedback Data (Admin Only)"):
    if os.path.exists("feedback.csv"):
        df = pd.read_csv("feedback.csv")
        st.dataframe(df)
    else:
        st.info("No feedback yet.")
