#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class DUDnCNN(NNRegressor):
    def __init__(self, D, C=64):
        super(DUDnCNN, self).__init__()
        self.D = D
        k = [0]
        k.extend([i for i in range(D//2)])
        k.extend([k[-1] for x in range(D//2, D+1)])
        l = [0 for x in range(D//2+1)]
        l.extend([i for i in range(D+1-(D//2+1))])
        l.append(l[-1])
        # holes and dilations for convolution layers
        holes = [2**(kl[0]-kl[1])-1 for kl in zip(k,l)] 
        dilations = [i+1 for i in holes] #dilation = holes + 1
        # convolution layers
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(3, C, 3, padding=dilations[0],dilation=dilations[0]))
        self.conv.extend([nn.Conv2d(C, C, 3, padding=dilations[i+1],dilation=dilations[i+1]) for i in range(D)])
        self.conv.append(nn.Conv2d(C, 3, 3, padding=dilations[-1],dilation=dilations[-1]))
        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for x in range(D)])
13
    def forward(self, x):
        D = self.D
        h = F.relu(self.conv[0](x))
        h_buff = []
        h_buff.append(h)
        for i in range(D//2 - 1):
            torch.backends.cudnn.benchmark = True
            h = self.conv[i+1](h)
            torch.backends.cudnn.benchmark = False
            h = F.relu(self.bn[i](h))
            h_buff.append(h)
        for i in range(D//2 - 1, D//2 + 1):
            torch.backends.cudnn.benchmark = True
            h = self.conv[i+1](h)
            torch.backends.cudnn.benchmark = False
            h = F.relu(self.bn[i](h))
        for i in range(D//2 + 1, D):
            j = i - (D//2 + 1) + 1
            torch.backends.cudnn.benchmark = True
            h = self.conv[i+1]((h + h_buff[-j]) / np.sqrt(2))
            torch.backends.cudnn.benchmark = False
            h = F.relu(self.bn[i](h))
        h = self.conv[D+1]((h+h_buff[0])/np.sqrt(2))
        y = h+x
        return y

