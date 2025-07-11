import torch
import torch.nn as nn

class CNNExpert(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4,8,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(8*10*12,32), nn.ReLU(),
            nn.Linear(32,n_classes))

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x)

class TransformerExpert(nn.Module):
    def __init__(self,n_classes,n_mfcc=40,seq_len=48,nhead=4,dim_feedforward=128):
        super().__init__()
        self.input_proj = nn.Linear(n_mfcc,dim_feedforward)
        layer = nn.TransformerEncoderLayer(dim_feedforward,nhead)
        self.encoder = nn.TransformerEncoder(layer,num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(dim_feedforward,n_classes)

    def forward(self,x):
        x = x.permute(2,0,1)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.permute(1,2,0)
        x = self.pool(x)
        x = x.squeeze(2)
        return self.classifier(x)

class VoiceMixtureModel(nn.Module):
    def __init__(self,n_classes,n_experts=2):
        super().__init__()
        self.experts = nn.ModuleList([
            CNNExpert(n_classes),
            TransformerExpert(n_classes)])
        self.gating = nn.Sequential(
            nn.Flatten(),
            nn.Linear(40*48,n_experts),
            nn.Softmax(dim=1))

    def forward(self,x):
        weights = self.gating(x)
        outputs = [e(x) for e in self.experts]
        stacked = torch.stack(outputs,dim=2)
        out = weights.unsqueeze(2) * stacked
        return out.sum(dim=2)

#very simple
#class VoiceCNN(nn.Module):
#    def __init__(self, n_classes):
#        super().__init__()
#        self.conv = nn.Sequential(
#            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1), nn.ReLU(),
#            nn.MaxPool2d(2))
#        self.fc = nn.Sequential(
#            nn.Flatten(),
#            nn.Linear(8 * 20 * 24, 32),
#            nn.ReLU(),
#            nn.Linear(32, n_classes))
#
#    def forward(self, x):
#        x = x.unsqueeze(1)
#        x = self.conv(x)
#        return self.fc(x)

#medium
#VoiceCNN(nn.Module):
#    def __init__(self, n_classes):
#        super().__init__()
#        self.conv = nn.Sequential(
#            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1), nn.ReLU(),
#            nn.MaxPool2d(2),
#            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1), nn.ReLU(),
#            nn.MaxPool2d(2))
#        self.fc = nn.Sequential(
#            nn.Flatten(),
#            nn.Dropout(0.5),
#            nn.Linear(8 * 10 * 12, 32), 
#            nn.ReLU(),
#            nn.Linear(32, n_classes))
#
#    def forward(self, x):
#        x = x.unsqueeze(1)
#        x = self.conv(x)
#        return self.fc(x)

