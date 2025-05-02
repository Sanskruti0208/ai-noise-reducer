# 🎧 AI Noise Reducer

A powerful noise reduction tool using **deep learning** and **pretrained audio separation models** (Demucs + CNN).  
Denoise your recorded or uploaded audio files in just a click with an elegant Streamlit interface.

---

## 🔥 Features

🎙️ Live Audio Recording or Upload .wav Files
Record audio directly in-browser or upload your own recordings.

🧠 Choose Denoising Model
Select between:

Demucs (Facebook AI Research)

Custom-trained CNN model

🌐 Streamlit-Based Interactive UI
Clean, responsive interface built with Streamlit – ready for deployment.

🌍 Multilingual Instructions
Users can choose their preferred language for instructions and feedback.

📊 Waveform Visualization and Comparison
Side-by-side before/after waveform plots using librosa and matplotlib.

💾 Automatic Saving of Outputs
Denoised audio and waveform images are automatically saved.

📣 User Feedback Support
Collect feedback directly through the interface for continuous improvement.

---

## 🗂️ Project Structure

```plaintext
ai_noise_reducer/
├── data/
│   └── recorded_audio/         # Raw user recordings or uploads
├── output/
│   └── enhanced_audio/         # Denoised audio files (Demucs + CNN)
├── output_imgs/
│   └── *.png                   # Waveform comparison images
├── separated/                  # Output from Demucs (auto-generated)
├── Notebooks/
│   └── model.py                # Custom CNN model architecture
├── model/
│   └── Denoising_model.pth     # Trained CNN weights
├── denoise_audio.py            # Core logic: record, denoise, plot
├── main.py                     # 🎨 Streamlit UI
├── requirements.txt            # Python dependencies
└── README.md                   # This file

---

## 🚀 Getting Started

1. Clone the repo

```bash
git clone https://github.com/yourusername/ai-noise-reducer.git
cd ai-noise-reducer

```
2. Install dependencies
Make sure you have Python 3.8 or higher.
```
pip install -r requirements.txt
```
3. Install Demucs
```
pip install demucs
streamlit run main.py
```
Optional: You can also install Demucs globally for CLI use.

🎛️ How to Use
🔧 Locally via Streamlit
```
streamlit run main.py
```
Choose:

Upload .wav file
or

Record live audio (5 seconds)

Then:

Apply noise reduction (Demucs / CNN)

Listen to output & view waveform comparison!



🧠 Custom CNN Model
Implemented in Notebooks/model.py

Trained on paired noisy/clean audio samples

Supports real-time inference using PyTorch

Replace Denoising_model.pth with your own trained weights if desired



📸 Sample Output

![image](https://github.com/user-attachments/assets/8a8d5c68-cab1-44e8-a603-907846d5829c)

🌐 Deployment:

Deploy easily on:

Streamlit Cloud ✅

Render.com

Hugging Face Spaces

Heroku (w/ backend support)

📜 License:

MIT License
Feel free to modify, fork, and use it for non-commercial projects. Contribute back if you enhance it! 🙌

💡 Author
Sanskruti Tidke
https://github.com/Sanskruti0208/



