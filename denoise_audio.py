import os
import sys
import subprocess
import sounddevice as sd
import soundfile as sf
import torch
import numpy as np
import matplotlib.pyplot as plt

# ========== PATH FIX FOR CUSTOM MODEL ==========
notebooks_path = os.path.join(os.getcwd(), 'Notebooks')
sys.path.append(notebooks_path)

from model import SimpleDenoisingCNN  # Ensure model.py exists in that path

# ========== CONFIG ==========
RECORD_DURATION = 5  # seconds
SR = 16000

data_dir = os.path.join("data", "recorded_audio")
output_dir = os.path.join("output", "enhanced_audio")
img_output_dir = os.path.join("output_imgs")
model_weights_path = os.path.join("model", "Denoising_model.pth")
demucs_model_name = "htdemucs"

os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(img_output_dir, exist_ok=True)

# ========== LOAD CUSTOM MODEL ==========
try:
    model = SimpleDenoisingCNN()
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Custom denoising model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# ========== UTILS ==========

def record_audio(filename, duration=RECORD_DURATION, sr=SR):
    print(f"🎤 Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    sf.write(filename, recording, sr)
    print(f"✅ Saved recording: {filename}")

def run_demucs(input_audio):
    print(f"\n🎧 Running Demucs on: {input_audio}")

    try:
        subprocess.run([
            "demucs", "--two-stems=vocals", "-n", demucs_model_name, input_audio
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during Demucs separation: {e}")
        return None

    filename = os.path.splitext(os.path.basename(input_audio))[0]
    demucs_out_path = os.path.join("separated", demucs_model_name, filename, "vocals.wav")

    if os.path.exists(demucs_out_path):
        final_path = os.path.join(output_dir, f"{filename}_denoised_demucs.wav")
        sf.write(final_path, *sf.read(demucs_out_path))
        print(f"✅ Denoised output saved to: {final_path}")
        return final_path
    else:
        print("❌ Demucs output not found.")
        return None

def plot_waveforms(original_audio, denoised_audio, sr, filename_prefix):
    time = np.linspace(0, len(original_audio) / sr, num=len(original_audio))

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, original_audio)
    plt.title("🔊 Original Noisy Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(time, denoised_audio)
    plt.title("🔇 Denoised Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    img_path = os.path.join(img_output_dir, f"{filename_prefix}_comparison.png")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

    print(f"🖼️  Waveform comparison saved at: {img_path}")
    return img_path

def run_custom_denoiser(input_audio):
    if model is None:
        print("⚠️ Custom model is not loaded. Skipping denoising.")
        return None, None

    print(f"\n🎧 Running Custom Denoiser on: {input_audio}")
    audio, sr = sf.read(input_audio)
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        denoised_tensor = model(audio_tensor)

    denoised_audio = denoised_tensor.squeeze().numpy()

    filename = os.path.splitext(os.path.basename(input_audio))[0]
    final_path = os.path.join(output_dir, f"{filename}_denoised_custom.wav")
    sf.write(final_path, denoised_audio, sr)
    print(f"✅ Custom denoised output saved to: {final_path}")

    img_path = plot_waveforms(audio, denoised_audio, sr, filename)

    return final_path, img_path
