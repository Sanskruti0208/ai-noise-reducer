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
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import queue
import wave

# Set up Streamlit page configuration
st.set_page_config(page_title="🎙️ AI Noise Reducer", layout="centered")

# Language selection
language = st.selectbox("Choose language for instructions:", ["English", "Spanish", "French", "Marathi", "Hindi"])
selected_text = texts[language]

st.title(selected_text["title"])
st.subheader(selected_text["subheader"])

# Input method
option = st.radio("Choose input method:", [selected_text["upload_audio"], selected_text["record_audio"]])

model_choice = st.selectbox(selected_text["choose_model"], ["Demucs", "Custom Denoiser"])
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

# Audio Recording Setup
audio_queue = queue.Queue()

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_queue.put(frame)
        return frame

def save_audio_from_queue(output_path, sample_rate=48000):
    frames = []
    while not audio_queue.empty():
        frame = audio_queue.get()
        audio = frame.to_ndarray().flatten().astype(np.int16)
        frames.append(audio)

    if not frames:
        return None

    audio_data = np.concatenate(frames)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return output_path

# Upload
if option == selected_text["upload_audio"]:
    uploaded_file = st.file_uploader("Upload your noisy audio (WAV)")
    if uploaded_file:
        if hasattr(uploaded_file, "name"):
            filename = uploaded_file.name
        else:
            filename = f"uploaded_{int(time.time())}.wav"

        os.makedirs("data/recorded_audio", exist_ok=True)
        input_path = os.path.join("data", "recorded_audio", filename)
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
                st.audio(demucs_out, format='audio/wav')
                st.subheader("Denoised Audio Waveform (Demucs)")
                plot_waveform(demucs_out, title="Denoised Audio (Demucs)")
            else:
                custom_out, img_path = run_custom_denoiser(input_path)
                hide_processing_status()
                st.success(selected_text["denoising_complete"].format(model="Custom Denoiser"))
                st.audio(custom_out, format='audio/wav')
                st.subheader("Denoised Audio Waveform (Custom Denoiser)")
                plot_waveform(custom_out, title="Denoised Audio (Custom Denoiser)")
                st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)

# Record
elif option == selected_text["record_audio"]:
    st.write("Press Start to record your voice using your microphone.")
    ctx = webrtc_streamer(
        key="audio",
        mode="sendonly",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False"},
    )

    if ctx.state.playing:
        if st.button("Save Recording"):
            os.makedirs("data/recorded_audio", exist_ok=True)
            filename = f"recorded_{int(time.time())}.wav"
            output_path = os.path.join("data", "recorded_audio", filename)
            result_path = save_audio_from_queue(output_path)

            if result_path:
                st.success("✅ Audio recorded successfully!")
                st.audio(result_path, format='audio/wav')
                st.subheader("Recorded Audio Waveform")
                plot_waveform(result_path, title="Recorded Audio")

                if st.button("Run Denoising"):
                    show_processing_status()
                    if model_choice == "Demucs":
                        output = run_demucs(result_path)
                    else:
                        output, img_path = run_custom_denoiser(result_path)
                    hide_processing_status()
                    st.success(selected_text["denoising_complete"].format(model=model_choice))
                    st.audio(output, format='audio/wav')
                    plot_waveform(output, title=f"Denoised Audio ({model_choice})")
                    if model_choice == "Custom Denoiser":
                        st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)
            else:
                st.error("⚠️ No audio was recorded. Please try again.")

# Feedback
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
