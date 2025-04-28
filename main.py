import os
import streamlit as st
from denoise_audio import run_custom_denoiser, run_demucs, record_audio

# Set up the page layout and title
st.set_page_config(page_title="🎙️ AI Noise Reducer", layout="centered")
st.title("🎧 AI Noise Reducer")
st.subheader("Denoise your recordings with AI!")

# Input method choice
option = st.radio("Choose input method:", ["Upload Audio", "Record Live Audio"])

# Create necessary directories if they don't exist
os.makedirs(os.path.join("data", "recorded_audio"), exist_ok=True)
os.makedirs("output/denoised", exist_ok=True)
os.makedirs("output_imgs", exist_ok=True)

# File upload method
if option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload your noisy audio (WAV)", type=["wav"])
    if uploaded_file:
        # Save the uploaded file to the appropriate directory
        input_path = os.path.join("data", "recorded_audio", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        # Show the uploaded audio file
        st.audio(input_path, format='audio/wav')

        # Run the denoising process
        if st.button("Run Denoising"):
            demucs_out = run_demucs(input_path)
            custom_out, img_path = run_custom_denoiser(input_path)

            st.success("✅ Denoising completed!")

            # Display the denoised audio and image comparison
            if custom_out:
                st.audio(custom_out, format='audio/wav')
                st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)

# Live recording method
elif option == "Record Live Audio":
    if st.button("Start Recording"):
        # Record live audio
        input_path = os.path.join("data", "recorded_audio", "live_record.wav")
        record_audio(input_path)
        st.audio(input_path, format='audio/wav')

        # Run denoising on the recorded file
        demucs_out = run_demucs(input_path)
        custom_out, img_path = run_custom_denoiser(input_path)

        st.success("✅ Denoising completed!")

        # Display the denoised audio and image comparison
        if custom_out:
            st.audio(custom_out, format='audio/wav')
            st.image(img_path, caption="Comparison (Noisy vs. Denoised)", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for sound clarity | [GitHub Repo](https://github.com/yourusername/ai-noise-reducer)")
