import torch.nn as nn
import torchvision as tv
vgg = tv.models.vgg16_bn(pretrained=True)
class VGG16Transfer(nn.Module):
    def __init__(self, n_class, fine_tuning=False):
        super(VGG16Transfer, self).__init__()
                
        self.n_class = n_class
        vgg = tv.models.vgg16_bn(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = fine_tuning
        self.features = vgg.features
        self.avgpool = vgg.avgpool
     
        self.relu    = nn.ReLU(inplace=True)

 
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        

    def forward(self, x):
        h = self.features(x)
       
        # Complete the forward function for the rest of the encoder
        #out_decoder=self.bn5(self.deconv5(self.bn4(self.deconv4(self.bn3(self.deconv3(self.bn2(self.deconv2(self.bn1(self.deconv1(x1))))))))))
        
        h = self.relu(self.bn1(self.deconv1(h)))
        h = self.relu(self.bn2(self.deconv2(h)))
        h = self.relu(self.bn3(self.deconv3(h)))
        h = self.relu(self.bn4(self.deconv4(h)))
        out_decoder = self.relu(self.bn5(self.deconv5(h)))

        # score = __(self.relu(__(out_encoder)))     
        # Complete the forward function for the rest of the decoder
        
        score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)
