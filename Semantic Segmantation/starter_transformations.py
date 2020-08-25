#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision import utils
from basic_fcn import *
from dataloader import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
from sklearn.metrics import jaccard_score as jsc



# In[2]:

transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(512,1024), torchvision.transforms.RandomAffine(6, translate=(0.1,0.9), scale=(1,5), shear=(-2,2)), torchvision.transforms.ToTensor()])
train_dataset = CityScapesDataset(csv_file='train.csv',transforms = transform)
val_dataset = CityScapesDataset(csv_file='val.csv')
test_dataset = CityScapesDataset(csv_file='test.csv')
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=4,
                          num_workers=0,
                          shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=4,
                          num_workers=0,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=1,
                          num_workers=0,
                          shuffle=True)


# In[3]:


#def getRequiredClasses(ouptuts,targets,labels):
    #required_cls=[0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
    


# In[4]:


def iou(pred, target,cls):
    
        
    #lbl = target[:,cls,:,:].cpu().numpy()
    #output = pred[:,cls,:,:].cpu().numpy()
    pred = pred.view(4,-1)
    target = target.view(4,-1)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    intersection=[]
    union=[]
    
    
    intersection = np.sum(np.logical_and(pred == cls, target == cls),axis=1)
    union = np.sum(np.logical_or(pred==cls, target ==cls),axis=1)
    
    iou = (intersection+1)/(union+1)
    iou = np.mean(iou)
    return iou


# In[5]:


def pixel_acc(pred, target):
#     print(pred.shape)
    cls = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
    pred = pred[:,cls,:,:]
    target = target[:,cls,:,:]
    
    matches = (pred == target)
    pixelacc = matches.sum().cpu().numpy()/matches.nelement()
  

#             matching_pixels =  !(pred.int() ^ target.int())
#     matching_pixels = np.array(matching_

    del pred, target
    torch.cuda.empty_cache()
    return pixelacc


# In[6]:


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.constant(m.bias, 0)
        
epochs     = 100
criterion = nn.CrossEntropyLoss()
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)
#fcn_model = torch.load('best_model')
optimizer = optim.Adam(fcn_model.parameters(), lr=8e-3)


# In[ ]:


use_gpu = torch.cuda.is_available()
if use_gpu:
    fcn_model = fcn_model.cuda()
    
def train():
    for epoch in range(epochs):
        ts = time.time()
        for iter, (X, tar, Y) in enumerate(train_loader):
            #print(epoch,iter)
            optimizer.zero_grad()

            if use_gpu:
                inputs = X.cuda()
                labels = Y.cuda()
            else:
                inputs, labels = X,Y

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            
            if iter % 50 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, 'best_model')

        print(val(epoch))
        fcn_model.train()


def val(epoch):
        fcn_model.eval()
        pixelAcc = 0
        count = 0
        ious = np.zeros((1,19))
        loss = 0
        with torch.no_grad():
            for idx, (X, tar, Y) in enumerate(val_loader):
#                 if idx == 10:
#                     break
                if(idx%50 ==0):
                    print(idx)
                if use_gpu:
                    inputs = X.cuda()
                    labels = Y.cuda()
                    targets = tar.cuda()
                else:
                    inputs, labels, targets = X,Y, tar
                outputs = fcn_model(inputs)
                
                outputs = F.softmax(outputs,dim=1)
                max_idx = torch.argmax(outputs, 1, keepdim=True)

                one_hot = torch.FloatTensor(outputs.shape)
                one_hot.zero_()
                one_hot = one_hot.cuda()
                one_hot.scatter_(1, max_idx, 1)
                
                temp = []
                loss+=criterion(outputs, labels).item()
                pixelAcc += pixel_acc(one_hot, targets)
                for i,cls in enumerate([7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]):

                    temp.append((iou(max_idx.int(),labels.int(),cls)))
                
                ious += temp
                del max_idx
                #print(ious/(idx+1))
        ious = np.asarray(ious) / (idx+1)
        loss /= idx*4
        pixelAcc /= idx*4
        print("Validation loss {}, pixelacc {}".format(loss, pixelAcc))
        #     [iou/count for iou in ious]
        return loss, pixelAcc, ious
    
def test():
    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    pass
    
if __name__ == "__main__":
#     print (val(0)) # show the accuracy before training
    train()


# In[ ]:





# In[ ]:




