import torch.nn as nn
import torch 

class CNN(nn.Module):
    def __init__(self, n_classes: int = 50) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
            
            nn.Conv2d(24, 48, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
            
            nn.Conv2d(48, 48, kernel_size=(5, 5)),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2400, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
