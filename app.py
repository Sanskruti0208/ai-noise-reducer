import streamlit as st
import os
import pandas as pd
import datetime
from denoise_audio import run_custom_denoiser, run_demucs, record_audio
import sounddevice as sd

st.set_page_config(page_title="🎙️ AI Noise Reducer", layout="centered")

st.title("🎧 AI Noise Reducer")
st.subheader("Denoise your recordings with AI!")

# ===== Language Selector (Multilingual Support Placeholder) =====
language = st.selectbox("🌐 Select Language", ["English", "हिन्दी", "Español", "Français"], index=0)

# ===== Model Selection =====
model_choice = st.selectbox("🤖 Choose Denoising Model", ["Demucs", "Custom CNN"])

# ===== Noise Reduction Strength Slider =====
strength = st.slider("🎚️ Noise Reduction Strength", min_value=0, max_value=100, value=80)

# ===== Input Method =====
option = st.radio("Choose input method:", ["Upload Audio", "Record Live Audio"])

is_streamlit_cloud = "streamlit" in os.environ.get("ST_WORKSPACE", "")

def process_audio(path_list):
    for input_path in path_list:
        st.audio(input_path, format='audio/wav')

        if st.button(f"Run Denoising on {os.path.basename(input_path)}", key=input_path):
            if model_choice == "Demucs":
                out_path = run_demucs(input_path)
                if out_path:
                    st.audio(out_path, format='audio/wav')
                    st.success("✅ Demucs denoising done.")
            else:
                out_path, img_path = run_custom_denoiser(input_path)
                if out_path:
                    st.audio(out_path, format='audio/wav')
                    st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)
                    st.success("✅ Custom CNN denoising done.")

# ===== Upload Flow =====
if option == "Upload Audio":
    uploaded_files = st.file_uploader("Upload WAV files", type=["wav"], accept_multiple_files=True)
    if uploaded_files:
        input_paths = []
        for file in uploaded_files:
            file_path = os.path.join("data", "recorded_audio", file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            input_paths.append(file_path)
        process_audio(input_paths)

# ===== Record Audio Flow =====
elif option == "Record Live Audio":
    if is_streamlit_cloud:
        st.warning("🚫 Live recording is only available locally.")
    else:
        try:
            if st.button("Start Recording"):
                input_path = os.path.join("data", "recorded_audio", "live_record.wav")
                record_audio(input_path)
                st.audio(input_path, format='audio/wav')
                process_audio([input_path])
        except sd.PortAudioError as e:
            st.error(f"❌ Microphone error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

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
