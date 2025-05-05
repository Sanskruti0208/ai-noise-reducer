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

# Model selection
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
    uploaded_file = st.file_uploader("Upload your noisy audio (WAV)", type=["wav"])
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

# Real-Time Record Audio using HTML/JS
elif option == selected_text["record_audio"]:
    st.subheader("🎙️ Record Audio Below")

    components.html(
        """
        <html>
        <body>
        <button onclick="startRecording()">Start Recording</button>
        <button onclick="stopRecording()">Stop & Send to App</button>
        <br /><br />
        <audio id="audioPlayback" controls></audio>
        <script>
            let mediaRecorder;
            let audioChunks = [];

            async function startRecording() {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = e => {
                    audioChunks.push(e.data);
                };
                mediaRecorder.start();
            }

            function stopRecording() {
                mediaRecorder.stop();
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        const base64String = reader.result.split(',')[1];
                        const message = JSON.stringify({ audio: base64String });
                        const textarea = window.parent.document.querySelector('textarea');
                        textarea.value = message;
                        textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    };
                    reader.readAsDataURL(audioBlob);
                    const audioUrl = URL.createObjectURL(audioBlob);
                    document.getElementById('audioPlayback').src = audioUrl;
                };
            }
        </script>
        </body>
        </html>
        """,
        height=300,
    )

    # Capture base64 audio from the frontend
    audio_json = st.text_area("Paste generated audio blob here (hidden field)", "", label_visibility="collapsed")

    if audio_json:
        try:
            audio_data = eval(audio_json)["audio"]
            audio_bytes = base64.b64decode(audio_data)
            recorded_path = "data/recorded_audio/recorded_from_html.wav"
            os.makedirs(os.path.dirname(recorded_path), exist_ok=True)

            with open(recorded_path, "wb") as f:
                f.write(audio_bytes)

            st.success("✅ Audio received and saved!")
            st.audio(recorded_path, format='audio/wav')
            st.subheader("Recorded Audio Waveform")
            plot_waveform(recorded_path, title="Recorded Audio")

            if st.button("Run Denoising"):
                show_processing_status()
                if model_choice == "Demucs":
                    output_path = run_demucs(recorded_path)
                else:
                    output_path, img_path = run_custom_denoiser(recorded_path)
                hide_processing_status()

                st.success(selected_text["denoising_complete"].format(model=model_choice))
                st.audio(output_path, format='audio/wav')
                plot_waveform(output_path, title=f"Denoised Audio ({model_choice})")
                if model_choice == "Custom Denoiser":
                    st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing audio: {e}")

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
