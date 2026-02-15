import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from src.models.cnn import CNN
from src.data.augment import AudioAugment
from src.config.config import ProcessingConfig, DatasetConfig, TrainConfig

config = ProcessingConfig()

class AudioPredictor:
    def __init__(
        self, 
        model_path: str, 
        config: ProcessingConfig = config, 
        device: str = 'cuda'
    ) -> None:
        self.config = config
        self.audio_dataset = AudioAugment()
        self.dataset_config = DatasetConfig()
        self.train_config = TrainConfig()
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> CNN:
        model = CNN(n_classes=len(self.dataset_config.esc50_labels))
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict) and "best_val_acc" in checkpoint:
            print(f"Model validation accuracy: {checkpoint['best_val_acc']:.4f}")
        model.to(self.device).eval()
        print("Model loaded successfully!\n")
        return model

    def _extract_patches(self, spectrogram: np.ndarray, hop: int) -> torch.Tensor:
        n_frames, _ = spectrogram.shape
        if n_frames < self.dataset_config.cnn_input_length:
            spectrogram = np.pad(spectrogram, ((0, self.dataset_config.cnn_input_length - n_frames), (0, 0)), mode="constant")
            n_frames = self.dataset_config.cnn_input_length

        patches = np.concatenate([
            spectrogram[s:s + self.dataset_config.cnn_input_length][np.newaxis, np.newaxis]
            for s in range(0, n_frames - self.dataset_config.cnn_input_length + 1, hop)
        ], axis=0)
        return torch.tensor(patches, dtype=torch.float32).to(self.device)
    
    def _run_inference(self, patches: torch.Tensor, batch_size: int) -> torch.Tensor:
        all_outputs = []
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                all_outputs.append(self.model(patches[i:i + batch_size]))
        return torch.cat(all_outputs, dim=0).mean(dim=0)

    def predict_class(self, spectrogram: np.ndarray, hop: int = 1) -> int:
        patches = self._extract_patches(spectrogram, hop)
        mean_activations = self._run_inference(patches, self.train_config.batch_size)
        return mean_activations.argmax().item()

    def predict_top_k(self, spectrogram: np.ndarray, hop: int = 1, top_k: int = 5):
        patches = self._extract_patches(spectrogram, hop)
        mean_logits = self._run_inference(patches, self.train_config.batch_size)
        probs = F.softmax(mean_logits, dim=0)
        top_probs, top_indices = torch.topk(probs, min(top_k, len(self.dataset_config.esc50_labels)))       
        return top_probs.cpu().numpy(), top_indices.cpu().numpy()

    def predict_file(self, audio_file: str, top_k: int = 5):
        spectrogram = np.array(self.audio_dataset._data_treatment_testing(audio_file)).squeeze()
        predicted_class = self.predict_class(spectrogram)
        top_probs, top_indices = self.predict_top_k(spectrogram, top_k=top_k)
        return predicted_class, top_probs, top_indices

