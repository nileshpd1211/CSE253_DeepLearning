#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class UDnCNN(NNRegressor):

    def __init__(self, D, C=64):
        super(UDnCNN, self).__init__()
        self.D = D
        
        # convolution layers
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(3, C, 3, padding=1))
        self.conv.extend([nn.Conv2d(C, C, 3, padding=1) for k in range(D)])
        self.conv.append(nn.Conv2d(C, 3, 3, padding=1))
        
        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for k in range(D)])
        

    def forward(self, x):
        D = self.D
        h = F.relu(self.conv[0](x))
        h_buff = []
        idx_buff = []
        shape_buff = []
        h_buff.append(h)
        for i in range(D//2-1):
            shape_buff.append(h.shape)
            h, idx = F.max_pool2d(F.relu(self.bn[i](self.conv[i+1](h))), 
                                  kernel_size=(2,2), return_indices=True)
            h_buff.append(h)
            idx_buff.append(idx)
        for i in range(D//2-1, D//2+1):
            h = F.relu(self.bn[i](self.conv[i+1](h)))
        for i in range(D//2+1, D):
            j = i - (D//2 + 1) + 1
            h = F.relu(self.bn[i](self.conv[i+1](F.max_unpool2d( (h+h_buff[-j])/np.sqrt(2),
                                idx_buff[-j], kernel_size=(2,2), output_size=shape_buff[-j]))))
        h = self.conv[D+1]((h + h_buff[0])/np.sqrt(2))
        y = h+x
        return y

