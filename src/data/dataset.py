import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Sequence 

from src.config.config import DatasetConfig

config = DatasetConfig()
            
class FullTFPatchesDataset(Dataset):
    def __init__(self, spectrograms: Sequence[np.ndarray], labels: Sequence[int], config: DatasetConfig = config) -> None:
        self.config = config
        self.patch_indices = []

        for spec_idx, spec in enumerate(spectrograms):
            n_frames = spec.shape[0]
            label = labels[spec_idx]

            if n_frames >= self.config.cnn_input_length:
                for start_frame in range(n_frames - self.config.cnn_input_length + 1):
                    self.patch_indices.append((spec_idx, start_frame, label))
            else:
                self.patch_indices.append((spec_idx, 0, label))
            
        self.spectrograms = spectrograms
    
    def __len__(self) -> int:
        return len(self.patch_indices)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        spec_idx, start_frame, label = self.patch_indices[idx]
        spec = self.spectrograms[spec_idx]
        
        n_frames = spec.shape[0]
        
        if n_frames >= self.config.cnn_input_length:
            patch = spec[start_frame:start_frame + self.config.cnn_input_length]
        else:
            pad = self.config.cnn_input_length - n_frames
            patch = np.pad(spec, ((0, pad), (0, 0)), mode='constant')
        
        patch = patch[np.newaxis, :, :]
        
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class RandomPatchDataset(Dataset):
    def __init__(self, spectrograms: Sequence[np.ndarray], labels: Sequence[int], config: DatasetConfig = config) -> None:
        self.config = config
        self.spectrograms = spectrograms
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.spectrograms[idx]
        label = self.labels[idx]
        n_frames = spec.shape[0]
        
        if n_frames >= self.config.cnn_input_length:
            start = np.random.randint(0, n_frames - self.config.cnn_input_length + 1)
            patch = spec[start:start + self.config.cnn_input_length]
        else:
            pad = self.config.cnn_input_length - n_frames
            patch = np.pad(spec, ((0, pad), (0, 0)), mode='constant')
        
        patch = patch[np.newaxis, :, :]
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)