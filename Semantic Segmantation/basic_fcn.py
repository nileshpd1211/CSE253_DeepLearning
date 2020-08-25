import torch.nn as nn

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        c1, c2, c3, c4, c5, c6, c7 = 32,64, 128, 256, 512, 512, 4096
        self.n_class = n_class
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
#         self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        
    def forward(self, x):
        h = self.conv_block1(x)
        h = self.conv_block2(h)
        h = self.conv_block3(h)
        h = self.conv_block4(h)
        
#         h = self.deconv1(h)
#         h = self.bn1(h)
#         h = self.relu(h)
        h = self.deconv2(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.deconv3(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.deconv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        
        score = self.classifier(h)

        return score  # size=(N, n_class, x.H/1, x.W/1)
