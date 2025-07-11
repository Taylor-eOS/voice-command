import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
from torch.utils.data import Dataset, DataLoader
import settings
from model import VoiceMixtureModel

class CommandDataset(Dataset):
    def __init__(self, data_dir, commands, transform=None, fixed_length=48):
        self.samples, self.labels = [], []
        self.transform = transform
        self.fixed_length = fixed_length
        for idx, cmd in enumerate(commands):
            for fname in os.listdir(os.path.join(data_dir, cmd)):
                if fname.endswith('.wav'):
                    self.samples.append(os.path.join(data_dir, cmd, fname))
                    self.labels.append(idx)
        all_feats = []
        for path in self.samples:
            wav, sr = torchaudio.load(path)
            if wav.size(0) > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            f = transform(wav).squeeze(0)[:, :fixed_length]
            if f.size(1) < fixed_length:
                pad = torch.zeros(f.size(0), fixed_length - f.size(1))
                f = torch.cat([f, pad], dim=1)
            all_feats.append(f.flatten())
        stacked = torch.stack(all_feats)
        self.mfcc_mean = stacked.mean(0)
        self.mfcc_std = stacked.std(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.samples[idx])
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        f = self.transform(wav).squeeze(0)
        if f.size(1) < self.fixed_length:
            pad = f.new_zeros(f.size(0), self.fixed_length - f.size(1))
            f = torch.cat([f, pad], dim=1)
        else:
            f = f[:, :self.fixed_length]
        f = f.flatten()
        f = (f - self.mfcc_mean) / (self.mfcc_std + 1e-6)
        return f.view(40, self.fixed_length), self.labels[idx]

def train(data_dir, commands, epochs=settings.EPOCHS, batch_size=settings.BATCH_SIZE):
    mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40, melkwargs={'n_fft': 400, 'hop_length': 160})
    ds = CommandDataset(data_dir, commands, transform=mfcc, fixed_length=48)
    torch.save({'mean': ds.mfcc_mean, 'std': ds.mfcc_std}, 'stats.pth')
    loader = DataLoader(ds, batch_size, shuffle=True)
    model = VoiceMixtureModel(len(commands))
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    for ep in range(epochs):
        model.train()
        total_loss, correct, seen = 0, 0, 0
        for feats, labels in loader:
            opt.zero_grad()
            out = model(feats)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * labels.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            seen += labels.size(0)
        avg_loss = total_loss / seen
        acc = correct / seen * 100
        print(f'Epoch {ep+1}/{epochs}  loss {avg_loss:.4f}  train-acc {acc:.1f}%')
    torch.save(model.state_dict(), 'voice_command.pth')

if __name__ == '__main__':
    train(settings.DATA_DIR, settings.COMMANDS)

