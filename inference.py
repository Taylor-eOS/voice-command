import sys
import os
import torch
import torchaudio
from model import VoiceMixtureModel
import settings

def predict(model_path, audio_path, commands, stats_path='stats.pth', fixed_length=48, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats = torch.load(stats_path, map_location='cpu')
    mean, std = stats['mean'], stats['std']
    model = VoiceMixtureModel(len(commands)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    wav, sr = torchaudio.load(audio_path)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr,16000)(wav)
    mfcc = torchaudio.transforms.MFCC(sample_rate=16000,n_mfcc=40,melkwargs={'n_fft':400,'hop_length':160})
    f = mfcc(wav).squeeze(0)
    if f.size(1) < fixed_length:
        pad = f.new_zeros(f.size(0),fixed_length - f.size(1))
        f = torch.cat([f, pad], dim=1)
    else:
        f = f[:, :fixed_length]
    flat = f.flatten()
    normalized = (flat - mean) / (std + 1e-6)
    inp = normalized.view(1,40,fixed_length).to(device)
    with torch.no_grad():
        logits = model(inp)
    probs = torch.softmax(logits,dim=1).squeeze()
    pred_idx = probs.argmax().item()
    confidence = probs.max().item()
    print(f"Prediction: {commands[pred_idx]} (confidence: {confidence:.3f})")
    all_probs = dict(zip(commands,probs.tolist()))
    print(f"All probabilities: {all_probs}")
    return commands[pred_idx]

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python inference.py <audio.wav>')
        sys.exit(1)
    result = predict('voice_command.pth',sys.argv[1],settings.COMMANDS)
    print(f"Final result: {result}")

