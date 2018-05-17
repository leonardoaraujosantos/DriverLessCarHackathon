import torch.nn as nn
import numpy as np

class CNNDriver(nn.Module):
    def __init__(self):
        super(CNNDriver, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=0, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, padding=0, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, padding=0, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(1152, 1164),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1164, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(10, 1),
            nn.Tanh()
        )
        #self.fc_out = nn.Linear(10,1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_out = size[0]  # number of rows
                fan_in = size[1]  # number of columns
                variance = np.sqrt(2.0 / (fan_in + fan_out))
                m.weight.data.normal_(0.0, variance)

    def forward(self, x):
        # The expected image size is 66x200
        conv_layers = self.conv_layers(x)

        # Reshape layer5 activation to a vector
        conv_layers_reshape = conv_layers.view(conv_layers.size(0), -1)

        fc_out = self.fc_layers(conv_layers_reshape)
        return fc_out