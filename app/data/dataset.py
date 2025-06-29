# data/dataset.py
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageWaveformDataset(Dataset):
    def __init__(self, waveform_dict, image_paths_dict, transform=None):
        self.samples = []
        self.transform = transform

        for c in range(120):
            c_key = f"condition_{c:03d}"
            for e in range(16):
                e_key = f"electrode_{e:02d}"
                waveforms = waveform_dict[c][e]
                image_paths = image_paths_dict[c_key][e_key]
                for i in range(waveforms.shape[0]):
                    self.samples.append((waveforms[i], image_paths[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        waveform, image_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        waveform = torch.tensor(waveform, dtype=torch.float32)  # shape (16, 48)
        return image, waveform
