import streamlit as st
import os
from denoise_audio import run_custom_denoiser, run_demucs, record_audio

st.set_page_config(page_title="🎙️ AI Noise Reducer", layout="centered")

st.title("🎧 AI Noise Reducer")
st.subheader("Denoise your recordings with AI!")

# Option to choose input method
option = st.radio("Choose input method:", ["Upload Audio", "Record Live Audio"])

# Check if the app is running on Streamlit Cloud
is_streamlit_cloud = "streamlit" in os.environ.get("ST_WORKSPACE", "")

if option == "Upload Audio":
    # Upload audio
    uploaded_file = st.file_uploader("Upload your noisy audio (WAV)", type=["wav"])
    if uploaded_file:
        input_path = os.path.join("data", "recorded_audio", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())
        st.audio(input_path, format='audio/wav')

        if st.button("Run Denoising"):
            demucs_out = run_demucs(input_path)
            custom_out, img_path = run_custom_denoiser(input_path)

            st.success("✅ Denoising completed!")

            if custom_out:
                st.audio(custom_out, format='audio/wav')
                st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)

elif option == "Record Live Audio":
    if is_streamlit_cloud:
        st.warning("🚫 Live recording is only available locally. Please use the upload option for cloud deployment.")
    else:
        # Start recording audio
        if st.button("Start Recording"):
            input_path = os.path.join("data", "recorded_audio", "live_record.wav")
            record_audio(input_path)
            st.audio(input_path, format='audio/wav')

            demucs_out = run_demucs(input_path)
            custom_out, img_path = run_custom_denoiser(input_path)

            st.success("✅ Denoising completed!")

            if custom_out:
                st.audio(custom_out, format='audio/wav')
                st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)

st.markdown("---")
st.markdown("Made with ❤️ for sound clarity | [GitHub Repo](https://github.com/yourusername/ai-noise-reducer)")
