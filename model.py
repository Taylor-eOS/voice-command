import torch.nn as nn

#very simple
class VoiceCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 20 * 24, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x)

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

