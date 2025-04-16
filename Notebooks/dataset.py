import os
import torch
import librosa
import torch.nn.functional as F
from torch.utils.data import Dataset

class NoiseReductionDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, sr=16000, target_length=50000):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.sr = sr
        self.target_length = target_length
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])
        self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])
        print(f"Found {len(self.noisy_files)} noisy files and {len(self.clean_files)} clean files.")

    def __len__(self):
        return min(len(self.noisy_files), len(self.clean_files))

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])

        noisy = self.load_audio(noisy_path)
        clean = self.load_audio(clean_path)

        noisy = F.pad(torch.tensor(noisy), (0, self.target_length - len(noisy)), 'constant', 0)
        clean = F.pad(torch.tensor(clean), (0, self.target_length - len(clean)), 'constant', 0)

        return noisy, clean

    def load_audio(self, filepath):
        audio, _ = librosa.load(filepath, sr=self.sr)
        return audio
