#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install --user pycocotools')


# In[31]:


import csv
from shutil import copyfile
from pycocotools.coco import COCO
from tqdm import tqdm
from torchvision import models
import data_loader
import torchvision
import torch.nn as nn
import torch
from build_vocab import *
from torch.nn.utils.rnn import pack_padded_sequence
import os
import io
import numpy as np
from random import shuffle
import copy
import torchvision.transforms as transforms



# In[3]:


import nltk
#nltk.download('punkt')


# In[4]:


#make directory and get annotations for training and testing
# !mkdir data
# !wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./data/
# !unzip ./data/captions_train-val2014.zip -d ./data/
# !rm ./data/captions_train-val2014.zip


# In[5]:


# !mkdir data/images
# !mkdir data/images/train
# !mkdir data/images/test


# In[6]:


coco = COCO('./data/annotations/captions_train2014.json')


# In[7]:


#get ids of training images
with open('TrainImageIds.csv', 'r') as f:
    reader = csv.reader(f)
    trainIds = list(reader)
    
trainIds = [int(i) for i in trainIds[0]]


# In[8]:


# for img_id in trainIds:
#     path = coco.loadImgs(img_id)[0]['file_name']
#     copyfile('/datasets/COCO-2015/train2014/'+path, './data/images/train/'+path)


# In[9]:


print(len(trainIds))


# In[10]:


cocoTest = COCO('./data/annotations/captions_val2014.json')


# In[11]:


with open('TestImageIds.csv', 'r') as f:
    reader = csv.reader(f)
    testIds = list(reader)
    
testIds = [int(i) for i in testIds[0]]


# In[12]:


# for img_id in testIds:
#     path = cocoTest.loadImgs(img_id)[0]['file_name']
#     copyfile('/datasets/COCO-2015/val2014/'+path, './data/images/test/'+path)


# In[13]:


# seqs=[]
# caption_ids = []
# for img_id in trainIds:
#     for i in range(0,len(coco.imgToAnns[img_id])):
#         caption_ids.append(coco.imgToAnns[img_id][i]['id'])
#         seqs.append(coco.imgToAnns[img_id][i]['caption'])

        
idx = copy.deepcopy(trainIds)
shuffle(idx)
m=np.floor(len(idx)*0.95)

tran_idx = idx[0:int(m)]
val_idx = idx[int(m)+1:]

traincaption_ids = []
valcaption_ids = []
caption_ids = []

for img_id in trainIds:
    for i in range(0,len(coco.imgToAnns[img_id])):
        caption_ids.append(coco.imgToAnns[img_id][i]['id'])

for img_id in tran_idx:
    for i in range(0,len(coco.imgToAnns[img_id])):
        traincaption_ids.append(coco.imgToAnns[img_id][i]['id'])

for img_id in val_idx:
    for i in range(0,len(coco.imgToAnns[img_id])):
        valcaption_ids.append(coco.imgToAnns[img_id][i]['id'])

resnet_model = models.resnet50(pretrained=True)



class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features



# Build the model
vocab = build_vocab('./data/annotations/captions_train2014.json',0)


# In[20]:


class DecoderRNN(nn.Module):
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=len(vocab), num_layers=1, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


# In[36]:


# def train_encoder():
# #     transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(224,224,pad_if_needed=True)])
#     transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(224,224,pad_if_needed=True)])
#     train_loader =data_loader.get_loader('./data/images/train/','./data/annotations/captions_train2014.json',caption_ids,vocab,transform,3,True,0)
#     criterion = nn.CrossEntropyLoss()
#     encoder = EncoderCNN(embed_size=256).cuda()
#     decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab), num_layers=1).cuda()
#     optimizer = torch.optim.Adam(encoder.parameters(), lr=8e-3)
#     loss_train = []
#     outputs = None
#     for epoch in range(0,7):
#         for i, (images, captions, lengths) in enumerate(train_loader):
#             images=images.cuda()
#             captions = captions.cuda()
#             encoded_output = encoder.forward(images)
#             outputs = decoder(encoded_output, captions, lengths)

#             targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
#             loss = criterion(outputs, targets)
#             loss_train.append(loss)
#             decoder.zero_grad()
#             encoder.zero_grad()
#             loss.backward()
#             optimizer.step()

            
#             # Print log info
#             f = open('log.txt','a')
#             if i % 5000 == 0:
#                 print('loss:',loss)

#                 sampled_ids = decoder.sample(encoded_output).cpu().numpy()
#                 for item,word_id in enumerate(sampled_ids):
#                     sampled_caption = []
#                     for word_id_2 in word_id:
#                         word = vocab.idx2word[word_id_2]
#                         sampled_caption.append(word)
#                         if word == '<end>':
#                             break
#                     sentence = ' '.join(sampled_caption)
# #                     print(item,'.',sentence)
#                     f.write('Loss: {}, iter: {}, epoch: {}, \n generated caption: {}. {} \n \n'.format(loss,i,epoch,item,str(sentence)))
#             f.close()
#             if i%50000 == 0: 
#                 torch.save(decoder.state_dict(), os.path.join(
#                        'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
#                 torch.save(encoder.state_dict(), os.path.join(
#                     'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
#         print('Epoch:',epoch)
        
        
def train_encoder():
#     transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(224,224,pad_if_needed=True)])
#     transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(512,512,pad_if_needed=True)])
    transform = transforms.Compose([ 
            transforms.Resize((512,512),2),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225)),
            transforms.ToPILImage()])
    train_loader = data_loader.get_loader('./data/images/train/','./data/annotations/captions_train2014.json',traincaption_ids,vocab,transform,64,True,0)
    val_loader = data_loader.get_loader('./data/images/train/','./data/annotations/captions_train2014.json',valcaption_ids,vocab,transform,64,True,0)

    
    
    criterion = nn.CrossEntropyLoss()
    encoder = EncoderCNN(embed_size=256).cuda()
    decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab), num_layers=1).cuda()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())

    optimizer = torch.optim.Adam(params, lr=1e-3)
    loss_train = []
    loss_val = []
    outputs = None
    f = open('train.txt','a')
    f_val = open('val.txt','a') 
    encoded_output = None
    for epoch in range(0,1000):
        for i, (images, captions, lengths) in enumerate(train_loader):
            if i%50 == 0:
                print('Training epoch {}, iteration {}'.format(epoch,i))
            images=images.cuda()
            captions = captions.cuda()
            
            encoded_output = encoder.forward(images)
            outputs = decoder(encoded_output, captions, lengths)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            loss = criterion(outputs, targets)
            
            if i%100 ==0:
                loss_train.append(loss.item())
            
            decoder.zero_grad()
            encoder.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            if i%100 == 0:
                f.write('Epoch: {}, Iter: {}, Train loss: {}\n'.format(epoch,i, loss_train))
                f.flush()
            
            encoder.train()
            decoder.train()
                  
            if i%200 ==0:
                with torch.no_grad():
                    val_loss=0
                    print('Validation for epoch {}'.format(epoch))
                    f_val.write('Epoch {}\n'.format(epoch))
                    for j,(images, captions, lengths) in enumerate(val_loader):
                        images=images.cuda()
                        captions = captions.cuda()
                        encoded_output = encoder.forward(images)
                        outputs = decoder(encoded_output, captions, lengths)
                        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                        val_loss += criterion(outputs, targets).item()
                        sampled_ids = decoder.sample(encoded_output).cpu().numpy()
                        if j==3:
                            for item in range(0,5):
                                t = np.random.randint(0,len(sampled_ids))
                                sampled_caption = []
                                actual_caption = []
                                for word_id_2 in sampled_ids[t]:
                                    word = vocab.idx2word[word_id_2]
                                    sampled_caption.append(word)
                                    if word == '<end>':
                                        break
                                for word_id_2 in captions[t].cpu().numpy():
                                    word = vocab.idx2word[word_id_2]
                                    actual_caption.append(word)
                                    if word == '<end>':
                                        break
                                sentence = ' '.join(sampled_caption)
                                actual_sentence = ' '.join(actual_caption)
                                f_val.write('Generated Caption: {} \n Actual Caption: {}\n\n'.format(str(sampled_caption),str(actual_caption)))
                                f_val.flush()
                    val_loss /=j
                    loss_val.append(val_loss)
                    f_val.write('Loss for epoch {}: {}'.format(epoch, loss_val))
                    f_val.flush()
            
            if i%500 == 0: 
                torch.save(decoder.state_dict(), os.path.join(
                       './models/decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    './models/encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
    f.close()
    f_val.close()
            
       

train_encoder()


# In[ ]:




