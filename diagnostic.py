import torch
import torchaudio
import os
import numpy as np
from collections import defaultdict
import settings

def analyze_dataset_distribution(data_dir, commands):
    distribution = defaultdict(int)
    file_lengths = []
    sample_rates = []
    for cmd in commands:
        cmd_path = os.path.join(data_dir, cmd)
        if os.path.exists(cmd_path):
            files = [f for f in os.listdir(cmd_path) if f.endswith('.wav')]
            distribution[cmd] = len(files)
            for fname in files[:5]:
                try:
                    wav, sr = torchaudio.load(os.path.join(cmd_path, fname))
                    file_lengths.append(wav.size(1))
                    sample_rates.append(sr)
                except:
                    continue
    print("Dataset distribution:")
    for cmd, count in distribution.items():
        print(f"  {cmd}: {count} samples")
    if file_lengths:
        print(f"\nAudio characteristics (first 5 samples per class):")
        print(f"  Length range: {min(file_lengths)} - {max(file_lengths)} samples")
        print(f"  Sample rates: {set(sample_rates)}")

def compare_features(model_path, train_audio_path, test_audio_path, commands, stats_path='stats.pth'):
    stats = torch.load(stats_path)
    mean, std = stats['mean'], stats['std']
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=40,
        melkwargs={'n_fft':400, 'hop_length':160})
    
    def process_audio(path, label):
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        f = mfcc(wav).squeeze(0)
        if f.size(1) < 48:
            pad = f.new_zeros(f.size(0), 48 - f.size(1))
            f = torch.cat([f, pad], dim=1)
        else:
            f = f[:, :48]
        flat = f.flatten()
        normalized = (flat - mean) / (std + 1e-6)
        print(f"\n{label}:")
        print(f"  Raw feature stats: mean={flat.mean():.4f}, std={flat.std():.4f}")
        print(f"  Range: [{flat.min():.4f}, {flat.max():.4f}]")
        print(f"  Normalized stats: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
        print(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
        return normalized.view(1, 40, 48)
    
    train_features = process_audio(train_audio_path, "Training sample")
    test_features = process_audio(test_audio_path, "Test sample")
    cosine_sim = torch.nn.functional.cosine_similarity(
        train_features.flatten(), test_features.flatten(), dim=0)
    print(f"\nFeature similarity: {cosine_sim:.4f}")

def test_model_predictions(model_path, commands, stats_path='stats.pth'):
    from model import VoiceCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VoiceCNN(len(commands)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    stats = torch.load(stats_path, map_location=device)
    mean, std = stats['mean'].to(device), stats['std'].to(device)
    print(f"\nNormalization stats:")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")
    print(f"  Std zeros: {(std < 1e-6).sum().item()} out of {len(std)}")

if __name__ == '__main__':
    analyze_dataset_distribution('data', settings.COMMANDS)
    test_model_predictions('voice_command.pth', settings.COMMANDS)

