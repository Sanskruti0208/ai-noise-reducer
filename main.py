import os
import streamlit as st
from denoise_audio import run_custom_denoiser, run_demucs, record_audio

# ====== PAGE SETUP ======
st.set_page_config(page_title="🎙️ AI Noise Reducer", layout="centered")
st.title("🎧 AI Noise Reducer")
st.subheader("Denoise your recordings with AI!")

# ====== CREATE NECESSARY DIRECTORIES ======
os.makedirs(os.path.join("data", "recorded_audio"), exist_ok=True)
os.makedirs("output/denoised", exist_ok=True)
os.makedirs("output_imgs", exist_ok=True)

# ====== INPUT METHOD ======
option = st.radio("Choose input method:", ["Upload Audio", "Record Live Audio"])

# ====== FILE UPLOAD ======
if option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload your noisy audio (WAV)", type=["wav"])
    
    if uploaded_file:
        input_path = os.path.join("data", "recorded_audio", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.audio(input_path, format='audio/wav')

        if st.button("Run Denoising"):
            with st.spinner("Processing with Demucs..."):
                demucs_out = run_demucs(input_path)

            with st.spinner("Processing with Custom Denoiser..."):
                custom_out, img_path = run_custom_denoiser(input_path)

            st.success("✅ Denoising completed!")

            if demucs_out:
                st.subheader("🎤 Demucs Denoised Output")
                st.audio(demucs_out, format='audio/wav')

            if custom_out:
                st.subheader("🎵 Custom Model Denoised Output")
                st.audio(custom_out, format='audio/wav')
                st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)

# ====== LIVE RECORDING ======
elif option == "Record Live Audio":
    if st.button("Start Recording"):
        input_path = os.path.join("data", "recorded_audio", "live_record.wav")
        record_audio(input_path)
        st.audio(input_path, format='audio/wav')

        with st.spinner("Processing with Demucs..."):
            demucs_out = run_demucs(input_path)

        with st.spinner("Processing with Custom Denoiser..."):
            custom_out, img_path = run_custom_denoiser(input_path)

        st.success("✅ Denoising completed!")

        if demucs_out:
            st.subheader("🎤 Demucs Denoised Output")
            st.audio(demucs_out, format='audio/wav')

        if custom_out:
            st.subheader("🎵 Custom Model Denoised Output")
            st.audio(custom_out, format='audio/wav')
            st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)

# ====== FOOTER ======
st.markdown("---")
st.markdown("Made with ❤️ for sound clarity | [GitHub Repo](https://github.com/yourusername/ai-noise-reducer)")
