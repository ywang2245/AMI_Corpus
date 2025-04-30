import torch
import torchaudio
import numpy as np

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=40, n_fft=400, hop_length=160):
        """
        Initialize the audio feature extractor.
        
        Args:
            sample_rate (int): Target sample rate for audio
            n_mels (int): Number of mel filterbanks
            n_fft (int): FFT window size
            hop_length (int): Hop length between frames
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Initialize mel spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # Initialize amplitude to DB transform
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def extract_features(self, audio_path):
        """
        Extract mel spectrogram features from audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            np.ndarray: Mel spectrogram features of shape [T, n_mels]
        """
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to decibels
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Remove batch dimension and transpose to [T, n_mels]
        mel_spec_db = mel_spec_db.squeeze(0).T
        
        return mel_spec_db.numpy() 