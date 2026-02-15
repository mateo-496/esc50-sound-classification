import tqdm
import librosa
import numpy as np
import os
import soundfile as sf
from typing import Optional

from src.config.config import ProcessingConfig

config = ProcessingConfig()

class AudioAugment:
    def __init__(self, config: ProcessingConfig = config) -> None: 
        self.config = config

    def _mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_fft=self.config.fft_size,
            hop_length=self.config.hop_size,
            win_length=self.config.frame_size,
            n_mels=self.config.n_bands,
            fmin=0,
            fmax=self.config.sample_rate / 2,
            window='hann'
        )
            
        mel_spectrogram_db = 10 * np.log10(mel_spec.T + 1e-10)
        max_db = mel_spectrogram_db.max()
        mel_spectrogram_db = mel_spectrogram_db - max_db

        return mel_spectrogram_db

    def _data_treatment_training(self, audio_path: str) -> tuple[list[np.ndarray], np.ndarray]:
        labels = []
        log_mel_spectrograms = []
        filenames = os.listdir(audio_path)
        
        for filename in tqdm.tqdm(filenames, desc="Processing audio files"):
            filename_splitted = filename.split("-")
            label = filename_splitted[-1].split(".")[0]
            label = label.split("_")[0]
            labels.append(int(label))
            
            file_path = os.path.join(audio_path, filename)
            audio, sr = librosa.load(file_path, sr=self.config.sample_rate)

            mel_spectrogram_db = self._mel_spectrogram(audio)
            log_mel_spectrograms.append(mel_spectrogram_db)
        
        return log_mel_spectrograms, np.array(labels)

    def _data_treatment_testing(self, file_path: str) -> list[np.ndarray]:
        audio, sr = librosa.load(file_path, sr=self.config.sample_rate)
        
        mel_spectrogram_db = self._mel_spectrogram(audio)

        return [mel_spectrogram_db]

    def _pad(self, audio: np.ndarray) -> np.ndarray:
        target_len = int(self.config.sample_rate * self.config.target_seconds)
        n = len(audio)

        if n < target_len:
            audio = np.pad(audio, (0, target_len - n), mode="constant")
        return audio

    def _time_stretch_augmentation(self, file_path: str, rate: float) -> np.ndarray:
        audio, _ = librosa.load(file_path, sr=self.config.sample_rate)
        audio_timestretch = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
        return self._pad(audio_timestretch)

    def _pitch_shift_augmentation(self, file_path: str, semitones: float) -> np.ndarray:
        audio, _ = librosa.load(file_path, sr=self.config.sample_rate)
        return librosa.effects.pitch_shift(audio.astype(np.float32), sr=self.config.sample_rate, n_steps=semitones)

    def _drc_augmentation(self, file_path: str, compression: float) -> np.ndarray:
        if compression == "musicstandard":   threshold_db=-20; ratio=2.0; attack_ms=5;  release_ms=50
        elif compression == "filmstandard":  threshold_db=-25; ratio=4.0; attack_ms=10; release_ms= 100
        elif compression == "speech":         threshold_db=-18; ratio=3.0; attack_ms=2;  release_ms= 40
        elif compression == "radio":          threshold_db=-15; ratio=3.5; attack_ms=1;  release_ms= 200

        audio, _ = librosa.load(file_path, sr=self.config.sample_rate)
        threshold = 10**(threshold_db / 20)

        attack_coeff  = np.exp(-1.0 / (0.001 * attack_ms * self.config.sample_rate))
        release_coeff = np.exp(-1.0 / (0.001 * release_ms * self.config.sample_rate))
        
        audio_filtered = np.zeros_like(audio)
        gain = 1.0
        
        for n in range(len(audio)):
            abs_audio = abs(audio[n])
            if abs_audio > threshold:
                desired_gain = (threshold / abs_audio) ** (ratio - 1)
            else:
                desired_gain = 1.0
            
            if desired_gain < gain:
                gain = attack_coeff * (gain - desired_gain) + desired_gain
            else:
                gain = release_coeff * (gain - desired_gain) + desired_gain
            
            audio_filtered[n] = audio[n] * gain

        return audio_filtered

    def _augment_dataset(self, audio_path: str, output_path: str, probability_list: list[float]) -> None:
        filenames = os.listdir(audio_path)

        p1, p2, p3 = probability_list
        os.makedirs(output_path, exist_ok=True)

        for filename in tqdm.tqdm(filenames, desc="Augmenting audio files"):        
            
            audio, _ = librosa.load(os.path.join(audio_path, filename), sr=self.config.sample_rate)
            # TS
            if np.random.rand() > p1:
                stretch_rates = [0.81, 0.93, 1.07, 1.23]
                stretch_rate = np.random.choice(stretch_rates)
                audio = self._time_stretch_augmentation(os.path.join(audio_path, filename), stretch_rate)
            # PS 
            if np.random.rand() > p2:
                semitones = [-3.5, -2.5, -2, -1, 1, 2.5, 3, 3.5]
                semitone = np.random.choice(semitones)
                audio = self._pitch_shift_augmentation(os.path.join(audio_path, filename), semitone)
            # DRC
            if np.random.rand() > p3:
                compressions = ["radio", "filmstandard", "musicstandard", "speech"]
                compression = np.random.choice(compressions)
                audio = self._drc_augmentation(os.path.join(audio_path, filename), compression)

            sf.write(os.path.join(output_path, filename), audio, self.config.sample_rate)

    def _create_augmented_datasets(self, input_path: str, output_path: str) -> None:
        probability_lists = self.config.augmentation_probability_lists
        for i, probability_list in enumerate(probability_lists):
            augmented_path = os.path.join(output_path, f"{i+1}")
            os.makedirs(augmented_path, exist_ok=True)
            self._augment_dataset(input_path, augmented_path, probability_list)

    def _create_log_mel(self, input_path: str, output_path: str) -> tuple[list[np.ndarray], np.ndarray]:
        directories = os.listdir(input_path)
        X, y = [], []

        for directory in directories:
            log_mels, labels = self._data_treatment_training(os.path.join(input_path, directory))
            X.extend(log_mels)
            y.extend(labels)
        
        X_array = np.empty(len(X), dtype=object)
        for i, spec in enumerate(X):
            X_array[i] = spec

        y = np.array(y)
        os.makedirs(output_path, exist_ok=True)
        
        np.save(os.path.join(output_path, "X.npy"), X_array, allow_pickle=True)
        np.save(os.path.join(output_path, 'y.npy'), y)
        return X, y

    def run(self, augment: bool = True, preprocess : bool = True) -> None:
        if augment:
            self._create_augmented_datasets(self.config.audio_path, self.config.augmented_path)
        if preprocess:
            self._create_log_mel(self.config.augmented_path, self.config.log_mel_path)