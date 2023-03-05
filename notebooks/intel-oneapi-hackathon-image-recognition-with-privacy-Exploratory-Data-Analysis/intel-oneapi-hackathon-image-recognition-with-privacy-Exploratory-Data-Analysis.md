# <a class="anchor" id="2"><div style="padding:28px;color:white;margin:0;font-size:48px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#8f0000;overflow:hidden;background:linear-gradient(100deg, navy, #251cab, #3eb4f4, #251cab);line-height:64px">Image Recognition using Computer Vision in eBPF platforms</div></a>

# <a class="anchor" id="2"><div style="padding:28px;color:white;margin:0;font-size:36px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:48px;background:linear-gradient(100deg, navy, #251cab, #3eb4f4, #251cab)">Demonstrating the Development of a Networking Packet transferring Vision Metadata</div></a>

# <a class="anchor" id="2"><div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:10px;background-color:#251cab;overflow:hidden;background:linear-gradient(100deg, navy, #251cab, #3eb4f4, #251cab)">1 | Introduction</div></a>

## Pre-import Libraries


```python
from IPython import display
```

<a class="anchor" id="2"><div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:10px;background-color:#251cab;overflow:hidden;background:linear-gradient(100deg, navy, #251cab, #3eb4f4, #251cab)">1.1 | Initial Data Analysis</div></a>
=========================

# Intel oneAPI Data Analytics Workflow


```python
display.Image("./images/Workflow.png")
```




    
![png](output_6_0.png)
    



# 1.1.1 Justification of Usage of NNCF

- Architecture of Intel AI Python Offering

## Intel oneAPI Data Analytics Toolkit

- Engineer Data
- Create ML anmd DL Models

## OpenVINO

- Deployment


```python
display.Image("./images/Intel-AI-Python-Offerings.png")
```




    
![png](output_8_0.png)
    



# 1.1.2 Import Libraries

**Type of Libraries to import**
- Neural Network Compressor (Intel)
- PyTorch (Text Recognition using CNN-LSTM)


```python
import nncf
import torch
from IPython import display
import torchvision
import textdistance
import lmdb
```

# 1.1.3 Download the Model

**CNN-LSTM Model**
- https://github.com/meijieru/crnn.pytorch

**A Simple OCR Model**
- https://github.com/Deepayan137/Adapting-OCR

**A Synthetic Data Generator for Generating Text Data**
- https://github.com/Belval/TextRecognitionDataGenerator


```python
!wget https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=0 -o models/crnn.pth
```

    --2023-02-25 00:23:10--  https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=0
    Resolving www.dropbox.com (www.dropbox.com)... 162.125.7.18, 2620:100:601b:18::a27d:812
    Connecting to www.dropbox.com (www.dropbox.com)|162.125.7.18|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: /s/raw/dboqjk20qjkpta3/crnn.pth [following]
    --2023-02-25 00:23:10--  https://www.dropbox.com/s/raw/dboqjk20qjkpta3/crnn.pth
    Reusing existing connection to www.dropbox.com:443.
    HTTP request sent, awaiting response... 302 Found
    Location: https://uc481e50d2abf48753241975a13b.dl.dropboxusercontent.com/cd/0/inline/B3I4Qg3dNmiQVL_ZI1ijbk2cnRlhfMUQ85bRuiRvRfJWj47YT52id0li3bx0Jws-PBIOCGgBetYUd5FFdEabEF8LVbJh5BmZbOyvBNOxN3656BtxhXrFbjW2HRC1cikQolQOSZ0NAWXOUyIjD8eV6nJ0TJeGw6wXeTpMw2SIrtJcgg/file# [following]
    --2023-02-25 00:23:10--  https://uc481e50d2abf48753241975a13b.dl.dropboxusercontent.com/cd/0/inline/B3I4Qg3dNmiQVL_ZI1ijbk2cnRlhfMUQ85bRuiRvRfJWj47YT52id0li3bx0Jws-PBIOCGgBetYUd5FFdEabEF8LVbJh5BmZbOyvBNOxN3656BtxhXrFbjW2HRC1cikQolQOSZ0NAWXOUyIjD8eV6nJ0TJeGw6wXeTpMw2SIrtJcgg/file
    Resolving uc481e50d2abf48753241975a13b.dl.dropboxusercontent.com (uc481e50d2abf48753241975a13b.dl.dropboxusercontent.com)... 162.125.7.15, 2620:100:601b:15::a27d:80f
    Connecting to uc481e50d2abf48753241975a13b.dl.dropboxusercontent.com (uc481e50d2abf48753241975a13b.dl.dropboxusercontent.com)|162.125.7.15|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: /cd/0/inline2/B3JtGylKm6P-bW9pgM2xIL0V6DKddxCxPVBYzcDD1LLoKwOlesYneuJVPIlRwy4wDfFoly5me3fziEtPHtGQ3wQ09OGJFWXMgigMcm13uQ5GcxFO9ZPyGiJJk29wTqj_7c6pltDWwjhpHLRW5ZImpUHiO-07FnwiyJ_QAMlIneY9qMiNt1n304agWcDXfu38IUH4T2M-lQtrVExrWHAoKJaSjPiEpqTO8t8rLa5LOc8XYIqhT6KhABlCVpabTf5AN9UUCpZjwaqRpYQ_WxZ14KaQZCRqZA_RTgLtr8xYfoY5SET5Sih5K4_ej0AYlq7Fatj7ByXRmT2SpObgCz6qs_P5HdlNziU_px5Z1TOyI8NeRpGcKecukjpm961dUhkNxyrCCPD3xuPOcRvv27y06OYLNwaqshSTnJG6SIQg-K30qg/file [following]
    --2023-02-25 00:23:11--  https://uc481e50d2abf48753241975a13b.dl.dropboxusercontent.com/cd/0/inline2/B3JtGylKm6P-bW9pgM2xIL0V6DKddxCxPVBYzcDD1LLoKwOlesYneuJVPIlRwy4wDfFoly5me3fziEtPHtGQ3wQ09OGJFWXMgigMcm13uQ5GcxFO9ZPyGiJJk29wTqj_7c6pltDWwjhpHLRW5ZImpUHiO-07FnwiyJ_QAMlIneY9qMiNt1n304agWcDXfu38IUH4T2M-lQtrVExrWHAoKJaSjPiEpqTO8t8rLa5LOc8XYIqhT6KhABlCVpabTf5AN9UUCpZjwaqRpYQ_WxZ14KaQZCRqZA_RTgLtr8xYfoY5SET5Sih5K4_ej0AYlq7Fatj7ByXRmT2SpObgCz6qs_P5HdlNziU_px5Z1TOyI8NeRpGcKecukjpm961dUhkNxyrCCPD3xuPOcRvv27y06OYLNwaqshSTnJG6SIQg-K30qg/file
    Reusing existing connection to uc481e50d2abf48753241975a13b.dl.dropboxusercontent.com:443.
    HTTP request sent, awaiting response... 200 OK
    Length: 33351680 (32M) [application/octet-stream]
    Saving to: ‘crnn.pth?dl=0’
    
    crnn.pth?dl=0       100%[===================>]  31.81M  3.99MB/s    in 8.0s    
    
    2023-02-25 00:23:20 (3.99 MB/s) - ‘crnn.pth?dl=0’ saved [33351680/33351680]
    


# Clone Git Repository


```python
!git clone https://github.com/Deepayan137/Adapting-OCR projects/Adapting-OCR/
```

    Cloning into 'projects'...
    remote: Enumerating objects: 198, done.[K
    remote: Counting objects: 100% (47/47), done.[K
    remote: Compressing objects: 100% (34/34), done.[K
    remote: Total 198 (delta 38), reused 13 (delta 13), pack-reused 151[K
    Receiving objects: 100% (198/198), 2.51 MiB | 2.52 MiB/s, done.
    Resolving deltas: 100% (96/96), done.


# Clone Git Repository


```python
!cd projects/ && git clone https://github.com/Belval/TextRecognitionDataGenerator
```

    Cloning into 'TextRecognitionDataGenerator'...
    remote: Enumerating objects: 1565, done.[K
    remote: Total 1565 (delta 0), reused 0 (delta 0), pack-reused 1565[K
    Receiving objects: 100% (1565/1565), 152.62 MiB | 3.75 MiB/s, done.
    Resolving deltas: 100% (652/652), done.
    Updating files: 100% (576/576), done.


# Clone Git Repository


```python
!git clone https://github.com/meijieru/crnn.pytorch projects/crnn.pytorch/
```

    Cloning into 'projects/crnn.pytorch'...
    remote: Enumerating objects: 92, done.[K
    remote: Counting objects: 100% (45/45), done.[K
    remote: Compressing objects: 100% (25/25), done.[K
    remote: Total 92 (delta 27), reused 23 (delta 20), pack-reused 47[K
    Unpacking objects: 100% (92/92), 44.12 KiB | 89.00 KiB/s, done.


# 1.1.4 Generate Synthetic Dataset on Text Recognition

#### **Package Used:**

* trdg package is used to generate text data with images
* About a 1000 images are created using this command, with specified height as 64 pixels
# trdg -c 1000 -w 5 -f 64


```python
!trdg -c 1000 -w 5 -f 64
```

# <a class="anchor" id="2"><div style="padding:28px;color:white;margin:0;font-size:36px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:48px;background:linear-gradient(100deg, navy, #251cab, #3eb4f4, #251cab)"><span style="color:lightgray">Business Case:</span> Federated Learning of Books Starting with Letters 'A - Z'</div></a>

# 1.1.5 Allocate the Privacy Budget

* The Privacy Budget refers to a list of Scanned taglines, from various corporations. In this scenario, the taglines are taken from each Cluster named - A, B, C, D, ... to Z
* The table below shows the Privacy Budget Allocated to Bucket A, Bucket B, etc, referring to the privacy preserved inside each Bucket or Cluster
* These Clusters together consist of 819 images

| Sl. No. | Alphabet | Cluster | Privacy Budget | Count |
|---------|----------|---------|----------------|-------|
| 1       | A        | A       | 0.542253536    | 51    |
| 2       | B        | B       | 0.178505244    | 31    |
| 3       | C        | C       | 0.080817933    | 74    |
| 4       | D        | D       | 0.26332368     | 38    |
| 5       | E        | E       | 0.912908671    | 26    |
| 6       | F        | F       | 0.799366428    | 29    |
| 7       | G        | G       | 0.860379176    | 18    |
| 8       | H        | H       | 0.58469011     | 36    |
| 9       | I        | I       | 0.596038276    | 33    |
| 10      | J        | J       | 0.460043453    | 5     |
| 11      | K        | K       | 0.920010075    | 4     |
| 12      | L        | L       | 0.573364902    | 23    |
| 13      | M        | M       | 0.417525399    | 38    |
| 14      | N        | N       | 0.487562981    | 32    |
| 15      | O        | O       | 0.408502215    | 32    |
| 16      | P        | P       | 0.545758955    | 80    |
| 17      | Q        | Q       | 0.604950905    | 10    |
| 18      | R        | R       | 0.252595279    | 34    |
| 19      | S        | S       | 0.662394842    | 92    |
| 20      | T        | T       | 0.532238687    | 36    |
| 21      | U        | U       | 0.692710454    | 43    |
| 22      | V        | V       | 0.061368844    | 23    |
| 23      | W        | W       | 0.974318856    | 26    |
| 24      | X        | X       | 0.366776829    | 1     |
| 25      | Y        | Y       | 0.173990203    | 3     |
| 26      | Z        | Z       | 0.303609772    | 1     |

## Estimated Allocated Privacy of All Combined Books into a Federated Model = 0.5032

# Model Inference

- Using nncf package
- Using a Text Recognition PyTorch Model, Trained here
- Quantize the model to verify the Inference Results, in ROC

# Model

- CRNN (CNN-LSTM) Model from PyTorch
- MONAI package is used to build PyTorch based Deep Neural Networks (DNNs)


```python
%%writefile models/crnn.py

import nncf
import torch
from IPython import display
import torchvision
import textdistance
import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
import random
import torch.nn as nn

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output
```

    Overwriting models/crnn.py


# Inference

- Pre-Trained Model is available as crnn.pth


```python
display.Image("/home/u186775/image-recognition-with-privacy/projects/TextRecognitionDataGenerator/sample_image/centro suprarenal rubbingstone chloro- semasiological_568.jpg")
```




    
![jpeg](output_28_0.jpg)
    



# <a class="anchor" id="2"><div style="padding:28px;color:black;margin:0;font-size:36px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:48px;background:linear-gradient(100deg, aqua, aqua, aqua, aqua)"><span style="color:darkgray">Use Case:</span> Text Recognition</div></a>

# 1.1.6 Inference of PyTorch Model for Text Recognition


```python
%%writefile inference.py

import sys

import os
import cv2
import pdb
import json
import math
import pickle
import logging
import warnings
from tqdm import *
import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torch.utils.data import random_split
from argparse import ArgumentParser
from copy import copy

import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import utils
from PIL import Image

import models.crnn as crnn

from src.data.pickle_dataset import PickleDataset
from src.data.synth_dataset import SynthDataset, SynthCollator

class Args():
    
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
            
class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

model_path = '/home/u186775/image-recognition-with-privacy/models/convolutional-reccurent-neural-network/crnn.pth'
img_path = '/home/u186775/image-recognition-with-privacy/projects/TextRecognitionDataGenerator/'
img_dir = 'sample_image'
# img_filename = 'amoebid equiparant prim-mannered hypercriticism nonsupplementary_301.jpg'
img_filename = 'centro suprarenal rubbingstone chloro- semasiological_568.jpg'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

args = Args(path=img_path, imgdir=img_dir, batch_size=32)
args.collate_fn = SynthCollator()
# args.alphabet = """Only thewigsofrcvdampbkuq.$A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%""" 
args.alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
args.nClasses = len(args.alphabet)

args.data = SynthDataset(args)

loader = torch.utils.data.DataLoader(args.data,
                batch_size=args.batch_size,
                collate_fn=args.collate_fn)
model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

print('new line %s' % model_path)

converter = utils.strLabelConverter(args.alphabet)
image = Image.open(os.path.join(img_path, img_dir, img_filename)).convert('L')
transformer = resizeNormalize((100, 32))
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print((raw_pred, sim_pred))
print('%-20s => %-20s' % (raw_pred, sim_pred))
```

    Overwriting inference.py



```python
%%writefile utils.py

#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections

class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img
```

    Overwriting utils.py


# 1.1.7 A qsub executable to submit to Intel DevCloud


```python
%%writefile inference.sh

#!/bin/bash

python3 /home/u186775/image-recognition-with-privacy/inference.py
```

    Overwriting inference.sh


# 1.1.8 Sending a Request to the Request Broker - qsub for a Compute Node


```python
cpu_job_id = !qsub inference.sh -l nodes=1:: -F "/home/u186775/image-recognition-with-privacy/projects/TextRecognitionDataGenerator out/ ./models/" -o log.out -e log.err
print(cpu_job_id[0])
```

    



```python
!qsub inference.sh -l nodes=1:xeon:gold6128:ppn=2 -F "/home/u186775/image-recognition-with-privacy/projects/TextRecognitionDataGenerator out/ /home/u186775/image-recognition-with-privacy/models/" -o logs/log.out -e logs/log.err
```

    2220181.v-qsvr-1.aidevcloud



```python
!qstat
```

    Job ID                    Name             User            Time Use S Queue
    ------------------------- ---------------- --------------- -------- - -----
    2219117.v-qsvr-1           ...ub-singleuser u186775         00:00:21 R jupyterhub     
    2220181.v-qsvr-1           inference.sh     u186775                0 R batch          



```python
!cat logs/log.out
```

    
    ########################################################################
    #      Date:           Mon 27 Feb 2023 04:18:26 AM PST
    #    Job ID:           2220181.v-qsvr-1.aidevcloud
    #      User:           u186775
    # Resources:           cput=75:00:00,neednodes=1:xeon:gold6128:ppn=2,nodes=1:xeon:gold6128:ppn=2,walltime=06:00:00
    ########################################################################
    
    loading pretrained model from /home/u186775/image-recognition-with-privacy/models/convolutional-reccurent-neural-network/crnn.pth
    new line /home/u186775/image-recognition-with-privacy/models/convolutional-reccurent-neural-network/crnn.pth
    ('ern--pe-n-----e--t-a-s---y', 'ernpenetasy')
    ern--pe-n-----e--t-a-s---y => ernpenetasy         
    
    ########################################################################
    # End of output for job 2220181.v-qsvr-1.aidevcloud
    # Date: Mon 27 Feb 2023 04:19:10 AM PST
    ########################################################################
    


## Error detected from submission of image to output 'ern--pe-n-----e--t-a-s---y => ernpenetasy'

# 1.1.9 Using Advanced OCR Model (PaddleOCR)


```python
from paddleocr import PaddleOCR, draw_ocr

ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = '/home/u186775/image-recognition-with-privacy/projects/TextRecognitionDataGenerator/sample_image/centro suprarenal rubbingstone chloro- semasiological_568.jpg'

result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
```

    [2023/02/27 07:15:20] ppocr DEBUG: Namespace(help='==SUPPRESS==', use_gpu=False, use_xpu=False, use_npu=False, ir_optim=True, use_tensorrt=False, min_subgraph_size=15, precision='fp32', gpu_mem=500, image_dir=None, page_num=0, det_algorithm='DB', det_model_dir='/home/u186775/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer', det_limit_side_len=960, det_limit_type='max', det_box_type='quad', det_db_thresh=0.3, det_db_box_thresh=0.6, det_db_unclip_ratio=1.5, max_batch_size=10, use_dilation=False, det_db_score_mode='fast', det_east_score_thresh=0.8, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_sast_score_thresh=0.5, det_sast_nms_thresh=0.2, det_pse_thresh=0, det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1, scales=[8, 16, 32], alpha=1.0, beta=1.0, fourier_degree=5, rec_algorithm='SVTR_LCNet', rec_model_dir='/home/u186775/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer', rec_image_inverse=True, rec_image_shape='3, 48, 320', rec_batch_num=6, max_text_length=25, rec_char_dict_path='/home/u186775/.local/lib/python3.9/site-packages/paddleocr/ppocr/utils/en_dict.txt', use_space_char=True, vis_font_path='./doc/fonts/simfang.ttf', drop_score=0.5, e2e_algorithm='PGNet', e2e_model_dir=None, e2e_limit_side_len=768, e2e_limit_type='max', e2e_pgnet_score_thresh=0.5, e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', e2e_pgnet_valid_set='totaltext', e2e_pgnet_mode='fast', use_angle_cls=True, cls_model_dir='/home/u186775/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer', cls_image_shape='3, 48, 192', label_list=['0', '180'], cls_batch_num=6, cls_thresh=0.9, enable_mkldnn=False, cpu_threads=10, use_pdserving=False, warmup=False, sr_model_dir=None, sr_image_shape='3, 32, 128', sr_batch_num=1, draw_img_save_dir='./inference_results', save_crop_res=False, crop_res_save_dir='./output', use_mp=False, total_process_num=1, process_id=0, benchmark=False, save_log_path='./log_output/', show_log=True, use_onnx=False, output='./output', table_max_len=488, table_algorithm='TableAttn', table_model_dir=None, merge_no_span_structure=True, table_char_dict_path=None, layout_model_dir=None, layout_dict_path=None, layout_score_threshold=0.5, layout_nms_threshold=0.5, kie_algorithm='LayoutXLM', ser_model_dir=None, re_model_dir=None, use_visual_backbone=True, ser_dict_path='../train_data/XFUND/class_list_xfun.txt', ocr_order_method=None, mode='structure', image_orientation=False, layout=True, table=True, ocr=True, recovery=False, use_pdf2docx_api=False, lang='en', det=True, rec=True, type='ocr', ocr_version='PP-OCRv3', structure_version='PP-StructureV2')
    [2023/02/27 07:15:23] ppocr DEBUG: dt_boxes num : 1, elapse : 0.7225682735443115
    [2023/02/27 07:15:23] ppocr DEBUG: cls num  : 1, elapse : 0.08473324775695801
    [2023/02/27 07:15:23] ppocr DEBUG: rec_res num  : 1, elapse : 0.4464852809906006
    [[[11.0, 17.0], [1204.0, 17.0], [1204.0, 51.0], [11.0, 51.0]], ('centro suprarenal rubbingstone chloro- semasiologica', 0.9524282813072205)]



```python
display.Image("/home/u186775/image-recognition-with-privacy/projects/TextRecognitionDataGenerator/sample_image/centro suprarenal rubbingstone chloro- semasiological_568.jpg")
```




    
![jpeg](output_43_0.jpg)
    



## In-Sample Single-Sample Accuracy = 1.0

## Out-of-Sample Single-Sample Accuracy = Cost of not recognizing 3 digits

**Descriptive Statistics**
==========================

* **Descriptive Statistics**
* **Inferential Statistics**

### __Panel Data__

<div class="lm-Widget p-Widget jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">

<table>
<thead>
<tr>
<th></th>
<th></th>
<th></th>
<th>PACKET INFORMATION</th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
</tr>
</thead>
<tbody><tr style="background-color:black;color:white">
<td>Sl. No.</td>
<td>Method</td>
<td>Type</td>
<td>Main</td>
<td>Category</td>
<td>Sub-Category</td>
<td>Trace1</td>
<td>Trace2</td>
<td>Content</td>
<td>Packet Type</td>
<td></td>
</tr>
<tr style="background-color:#fff5ce;color:black">
<td>1</td>
<td>OCR Text Recognition</td>
<td>VEXF</td>
<td>Passport Number</td>
<td>Date Of Birth</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>LEVEL 2  WiFi FRAME</td>
<td>SCAN OF THE PASSPORT DOCUMENT</td>
</tr>
<tr style="background-color:#fff5ce;color:black">
<td>2</td>
<td>OCR Text Recognition</td>
<td>VEXF</td>
<td></td>
<td></td>
<td>Nationality</td>
<td>Place of Issue</td>
<td>-</td>
<td>-</td>
<td></td>
<td></td>
</tr>
<tr style="background-color:#fff5ce;color:black">
<td>3</td>
<td>OCR Text Recognition</td>
<td>VEXF</td>
<td></td>
<td>Date of Birth</td>
<td>-</td>
<td>-</td>
<td>Name</td>
<td>-</td>
<td></td>
<td></td>
</tr>
<tr style="background-color:#fff5ce;color:black">
<td>4</td>
<td>OCR Text Recognition</td>
<td>VEXF</td>
<td>Passport Number</td>
<td>-</td>
<td>Nationality</td>
<td>-</td>
<td></td>
<td>-</td>
<td></td>
<td></td>
</tr>
<tr style="background-color:#fff5ce;color:black">
<td>5</td>
<td>OCR Text Recognition</td>
<td>VEXF</td>
<td></td>
<td></td>
<td></td>
<td>Place of Issue</td>
<td></td>
<td>-</td>
<td></td>
<td></td>
</tr>
<tr style="background-color: #ffd8ce;color:black">
<td>6</td>
<td>Internet Of Things</td>
<td>VEXF</td>
<td>EPC Number</td>
<td>Manufacturer</td>
<td>Time of Purchase</td>
<td>Maintenance Cycle</td>
<td>GTIN Number</td>
<td>-</td>
<td>ETHERNET FRAME</td>
<td>IoT INFORMATION</td>
</tr>
<tr style="background-color: #ffd8ce;color:black">
<td>7</td>
<td>Object Detection</td>
<td>VEXF</td>
<td>Number Plate</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td></td>
<td>STREET DATASET</td>
</tr>
<tr style="background-color: #ffd8ce;color:black">
<td>8</td>
<td>Packet Transfer</td>
<td>VEXF</td>
<td>Name</td>
<td>Date of Birth</td>
<td>-</td>
<td>-</td>
<td></td>
<td>-</td>
<td></td>
<td>PATIENT RECORDS</td>
</tr>
<tr style="background-color:#ffd8ce;color:black">
<td>9</td>
<td>Packet Transfer</td>
<td>VEXF</td>
<td>-</td>
<td>-</td>
<td>Consultant</td>
<td>Hospital</td>
<td>-</td>
<td>-</td>
<td></td>
<td>PATIENT RECORDS</td>
</tr>
<tr style="background-color:#ffd8ce;color:black">
<td>10</td>
<td>Packet Transfer</td>
<td>VEXF</td>
<td>Person Name</td>
<td>Event Location</td>
<td>Event Date</td>
<td>Delegate Name</td>
<td>Event Title</td>
<td>-</td>
<td></td>
<td>CALENDAR INFO.</td>
</tr>
<tr style="background-color:#dee7e5;color:black">
<td>11</td>
<td>Face Recognition</td>
<td>HTTP over VEXF</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Face Image</td>
<td>HTTP  Over VEXF</td>
<td>FACE RECOGNITION</td>
</tr>
<tr style="background-color:#dee7e5;color:black">
<td>12</td>
<td>Image Recognition</td>
<td>HTTP over VEXF</td>
<td>E-Tag Code</td>
<td>E-Tag Code</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Content</td>
<td></td>
<td>INFO. RETRIEVAL</td>
</tr>
<tr style="background-color:#dee7e5;color:black">
<td>13</td>
<td>Image Recognition</td>
<td>HTTP over VEXF</td>
<td>App ID</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td></td>
<td>E-DISCOVERY</td>
</tr>
<tr style="background-color:#dee7e5;color:black">
<td>14</td>
<td>Image Recognition</td>
<td>HTTP over VEXF</td>
<td>Date &amp; Time</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td></td>
<td>E-DISCOVERY</td>
</tr>
<tr style="background-color:#dee7e5;color:black">
<td>15</td>
<td>Image Recognition</td>
<td>HTTP over VEXF</td>
<td>Name</td>
<td>Size</td>
<td>Location</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td></td>
<td>E-DISCOVERY</td>
</tr>
</tbody></table>
</div>

# <a class="anchor" id="2"><div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:10px;background-color:#251cab;overflow:hidden;background:linear-gradient(100deg, navy, #251cab, #3eb4f4, #251cab)">2 | Exploratory Data Analysis</div></a>

# <a class="anchor" id="2"><div style="padding:28px;color:white;margin:0;font-size:36px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:48px;background:linear-gradient(100deg, navy, #251cab, #3eb4f4, #251cab)"><span style="color:lightgray">Business Case:</span> An Internet Share of the Scanned Copy of your Passport Across the Network</div></a>

# <a class="anchor" id="2"><div style="padding:20px;color:white;margin:0;font-size:80%;text-align:left;display:fill;border-radius:10px;background-color:#251cab;overflow:hidden;background:linear-gradient(100deg, maroon, brown, darkorange, darkorange)">Research Question 1 (RQ1):</div></a>
-----------------------------------

### **What meaningful data / information would you use to represent a single page containing text?**


```python
display.Image("./inference_images/sample_passport_computer_vision.jpg")
```




    
![jpeg](output_50_0.jpg)
    



### **Extract Text from Text Recognition**




```python
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = '/home/u186775/image-recognition-with-privacy/inference_images/sample_passport_computer_vision.jpg'

result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
```

    [2023/02/27 06:05:31] ppocr DEBUG: Namespace(help='==SUPPRESS==', use_gpu=False, use_xpu=False, use_npu=False, ir_optim=True, use_tensorrt=False, min_subgraph_size=15, precision='fp32', gpu_mem=500, image_dir=None, page_num=0, det_algorithm='DB', det_model_dir='/home/u186775/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer', det_limit_side_len=960, det_limit_type='max', det_box_type='quad', det_db_thresh=0.3, det_db_box_thresh=0.6, det_db_unclip_ratio=1.5, max_batch_size=10, use_dilation=False, det_db_score_mode='fast', det_east_score_thresh=0.8, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_sast_score_thresh=0.5, det_sast_nms_thresh=0.2, det_pse_thresh=0, det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1, scales=[8, 16, 32], alpha=1.0, beta=1.0, fourier_degree=5, rec_algorithm='SVTR_LCNet', rec_model_dir='/home/u186775/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer', rec_image_inverse=True, rec_image_shape='3, 48, 320', rec_batch_num=6, max_text_length=25, rec_char_dict_path='/home/u186775/.local/lib/python3.9/site-packages/paddleocr/ppocr/utils/en_dict.txt', use_space_char=True, vis_font_path='./doc/fonts/simfang.ttf', drop_score=0.5, e2e_algorithm='PGNet', e2e_model_dir=None, e2e_limit_side_len=768, e2e_limit_type='max', e2e_pgnet_score_thresh=0.5, e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', e2e_pgnet_valid_set='totaltext', e2e_pgnet_mode='fast', use_angle_cls=True, cls_model_dir='/home/u186775/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer', cls_image_shape='3, 48, 192', label_list=['0', '180'], cls_batch_num=6, cls_thresh=0.9, enable_mkldnn=False, cpu_threads=10, use_pdserving=False, warmup=False, sr_model_dir=None, sr_image_shape='3, 32, 128', sr_batch_num=1, draw_img_save_dir='./inference_results', save_crop_res=False, crop_res_save_dir='./output', use_mp=False, total_process_num=1, process_id=0, benchmark=False, save_log_path='./log_output/', show_log=True, use_onnx=False, output='./output', table_max_len=488, table_algorithm='TableAttn', table_model_dir=None, merge_no_span_structure=True, table_char_dict_path=None, layout_model_dir=None, layout_dict_path=None, layout_score_threshold=0.5, layout_nms_threshold=0.5, kie_algorithm='LayoutXLM', ser_model_dir=None, re_model_dir=None, use_visual_backbone=True, ser_dict_path='../train_data/XFUND/class_list_xfun.txt', ocr_order_method=None, mode='structure', image_orientation=False, layout=True, table=True, ocr=True, recovery=False, use_pdf2docx_api=False, lang='en', det=True, rec=True, type='ocr', ocr_version='PP-OCRv3', structure_version='PP-StructureV2')
    [2023/02/27 06:05:32] ppocr DEBUG: dt_boxes num : 35, elapse : 0.19899320602416992
    [2023/02/27 06:05:32] ppocr DEBUG: cls num  : 35, elapse : 0.19606757164001465
    [2023/02/27 06:05:35] ppocr DEBUG: rec_res num  : 35, elapse : 2.7051517963409424
    [[[23.0, 7.0], [486.0, 7.0], [486.0, 19.0], [23.0, 19.0]], ('UNITED KINGDOM OFGREAT BRITAIN AND NORTHERNIRELAND', 0.9573935866355896)]
    [[[88.0, 19.0], [207.0, 19.0], [207.0, 33.0], [88.0, 33.0]], ('PASSPORTType/pe', 0.8358199596405029)]
    [[[258.0, 20.0], [302.0, 20.0], [302.0, 31.0], [258.0, 31.0]], ('Code/Code-', 0.9174631834030151)]
    [[[257.0, 30.0], [284.0, 30.0], [284.0, 45.0], [257.0, 45.0]], ('GBR', 0.994743287563324)]
    [[[368.0, 21.0], [427.0, 21.0], [427.0, 31.0], [368.0, 31.0]], ('Passport NoPas', 0.843869686126709)]
    [[[370.0, 30.0], [458.0, 30.0], [458.0, 43.0], [370.0, 43.0]], ('107185703', 0.9970763325691223)]
    [[[423.0, 22.0], [450.0, 21.0], [450.0, 29.0], [423.0, 30.0]], ('sport No', 0.715143084526062)]
    [[[90.0, 33.0], [181.0, 29.0], [182.0, 42.0], [90.0, 46.0]], ('PASSEPORTP', 0.9767441749572754)]
    [[[171.0, 56.0], [328.0, 56.0], [328.0, 69.0], [171.0, 69.0]], ('UNITED-KINGDOM-FIVE', 0.9927221536636353)]
    [[[193.0, 48.0], [228.0, 48.0], [228.0, 55.0], [193.0, 55.0]], ('neNom(', 0.7733447551727295)]
    [[[173.0, 72.0], [253.0, 72.0], [253.0, 82.0], [173.0, 82.0]], ('Given name/Prnoms(2)', 0.8413727879524231)]
    [[[171.0, 82.0], [263.0, 82.0], [263.0, 95.0], [171.0, 95.0]], ('JODIE PIPPA', 0.9578777551651001)]
    [[[171.0, 107.0], [296.0, 107.0], [296.0, 120.0], [171.0, 120.0]], ('BRITISH CITIZEN', 0.9585227966308594)]
    [[[172.0, 122.0], [278.0, 122.0], [278.0, 132.0], [172.0, 132.0]], ('Date ol birth/Date de nance)', 0.7660800218582153)]
    [[[172.0, 133.0], [288.0, 133.0], [288.0, 146.0], [172.0, 146.0]], ('17JAN/JAN 85', 0.9580116271972656)]
    [[[172.0, 148.0], [297.0, 148.0], [297.0, 160.0], [172.0, 160.0]], ('Sex/Sexe Pace ol birthvUeu de ', 0.6982715129852295)]
    [[[172.0, 159.0], [181.0, 159.0], [181.0, 170.0], [172.0, 170.0]], ('F', 0.997626006603241)]
    [[[223.0, 157.0], [275.0, 157.0], [275.0, 171.0], [223.0, 171.0]], ('LONDON', 0.99690181016922)]
    [[[315.0, 151.0], [330.0, 151.0], [330.0, 157.0], [315.0, 157.0]], ('(6', 0.7919881939888)]
    [[[170.0, 181.0], [289.0, 182.0], [289.0, 199.0], [170.0, 198.0]], ('31 JAN /JAN 06', 0.9329398274421692)]
    [[[173.0, 174.0], [200.0, 174.0], [200.0, 183.0], [173.0, 183.0]], ('Datef', 0.8085024952888489)]
    [[[210.0, 174.0], [282.0, 174.0], [282.0, 183.0], [210.0, 183.0]], ('Date de dance 7', 0.7774632573127747)]
    [[[338.0, 183.0], [372.0, 183.0], [372.0, 198.0], [338.0, 198.0]], ('UKPA', 0.9974774718284607)]
    [[[389.0, 176.0], [402.0, 176.0], [402.0, 182.0], [389.0, 182.0]], ('408', 0.5523099899291992)]
    [[[172.0, 209.0], [288.0, 209.0], [288.0, 223.0], [172.0, 223.0]], ('31 JAN /JAN16', 0.9436445832252502)]
    [[[173.0, 200.0], [282.0, 200.0], [282.0, 209.0], [173.0, 209.0]], ("Date ol expiry/Date d'expiration(9)", 0.8709656000137329)]
    [[[337.0, 200.0], [389.0, 200.0], [389.0, 209.0], [337.0, 209.0]], ("Holder's gna", 0.6801825165748596)]
    [[[428.0, 200.0], [474.0, 200.0], [474.0, 209.0], [428.0, 209.0]], ('Ju titulie10', 0.6934114098548889)]
    [[[354.0, 223.0], [462.0, 223.0], [462.0, 246.0], [354.0, 246.0]], ('J-Oladbabus', 0.6953461170196533)]
    [[[26.0, 274.0], [473.0, 274.0], [473.0, 287.0], [26.0, 287.0]], ('P<GBRUNITED<KINGDOM<FIVE<<JODIE<PIPPA<<<<<<<', 0.9687700271606445)]
    [[[25.0, 298.0], [474.0, 297.0], [474.0, 313.0], [25.0, 314.0]], ('1071857032GBR8501178F1601312<<<<<<<<<<<<<<02', 0.9773078560829163)]


### **Show the List of Identifiers or Words within a Passport Document**


```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./inference_images/sample_passport_computer_vision.jpg", cv2.IMREAD_UNCHANGED)

# passport number
main = [[[370, 30], [458, 30], [458, 43], [370, 43]], ('107185703', 0.9970763325691223)]

# date of birth
category = [[[172, 133], [288, 133], [288, 146], [172, 146]], ('17JAN/JAN 85', 0.9580116271972656)]

# nationality
sub_category = [[[171, 107], [296, 107], [296, 120], [171, 120]], ('BRITISH CITIZEN', 0.9585227966308594)]

# place of issue
trace1 = [[[223, 157], [275, 157], [275, 171], [223, 171]], ('LONDON', 0.99690181016922)]

# name
trace2 = [[[171, 82], [263, 82], [263, 95], [171, 95]], ('JODIE PIPPA', 0.9578777551651001)]

# annotating passport number
img = cv2.rectangle(img, tuple(main[0][0]), tuple(main[0][2]), (0,0,255), 2)

# annotating date of birth
img = cv2.rectangle(img, tuple(category[0][0]), tuple(category[0][2]), (0,255,255), 2)

# annotating nationality
img = cv2.rectangle(img, tuple(sub_category[0][0]), tuple(sub_category[0][2]), (0,255,0), 2)

# annotating place of issue
img = cv2.rectangle(img, tuple(trace1[0][0]), tuple(trace1[0][2]), (255,0,0), 2)

# annotating the name
img = cv2.rectangle(img, tuple(trace2[0][0]), tuple(trace2[0][2]), (200,0,0), 2)

plt.imshow(img)
plt.show()
```


    
![png](output_54_0.png)
    


### **Ordered Histogram of Relevance**


```python
import numpy as np

priv_budget = np.array([12, 6, 3, 1, 0.5])
priv_budget /= priv_budget.sum()

fig, ax = plt.subplots()

labels = ['Passport Number', 'Date.Of.Birth', 'Nationality', 'Place of Issue', 'Name']
ax.bar(labels, priv_budget, color=['blue', 'aqua', 'green', 'red', 'brown'])
ax.set_xlabel("Privacy-Preserved Attributes of a Passport", fontsize=11)
ax.set_ylabel("Relevance of Privacy", fontsize=11)
ax.set_xticklabels(labels, rotation=45)
fig.show()
```


    
![png](output_56_0.png)
    


### **A Representation Problem: Heap-Structure Encoding in an Image**


```python
from heap_viz.visualize import Node, drawTree2
```


```python
priv_budget
```




    array([0.53333333, 0.26666667, 0.13333333, 0.04444444, 0.02222222])




```python
# tree1 :: Tree Int
tree1 = Node(0.53333333)([
    Node(0.26666667)([
        Node(0.04444444)([]),
        Node(0.02222222)([])
    ]),
    Node(0.13333333)([])
])

print(
    '\n\n'.join([
        'Centered parents with nodeless lines pruned out:',
        drawTree2(False)(True)(
            tree1
        )
    ])
)
```

    Centered parents with nodeless lines pruned out:
    
                             ┌ 0.04444444 
                ┌ 0.26666667 ┤
     0.53333333 ┤            └ 0.02222222 
                └ 0.13333333 


### **Create a Heap-Structure**

#### **Heap Class Definition**
-------------------------------


```python
# First let us complete a minheap data structure.
# Please complete missing parts below.
class MaxHeap:
    def __init__(self):
        self.H = [None]
        
    def size(self):
        return len(self.H)-1
    
    def __repr__(self):
        return str(self.H[1:])
        
    def satisfies_assertions(self):
        for i in range(2, len(self.H)):
            assert self.H[i] <= self.H[i//2],  f'Maxheap property fails at position {i//2}, parent elt: {self.H[i//2]}, child elt: {self.H[i]}'
    
    def max_element(self):
        return self.H[1]
```

##### **The Bubble Up Method**
-------------------------------


```python
## bubble_up function at index
## WARNING: this function has been cut and paste for the next problem as well 
def bubble_up(self, index):
    # your code here
    assert index >= 1
    if index == 1: 
        return 
    parent_index = index // 2
    if self.H[parent_index] > self.H[index]:
        return 
    else:
        self.H[parent_index], self.H[index] = self.H[index], self.H[parent_index]
        self.bubble_up(parent_index)
```

##### **The Bubble Down Method**
-------------------------------


```python
## bubble_down function at index
## WARNING: this function has been cut and paste for the next problem as well 
def bubble_down(self, index):
    # your code here
    assert index >= 1 and index < len(self.H)
    lchild_index = 2 * index
    rchild_index = 2 * index + 1
    # set up the value of left child to the element at that index if valid, or else make it +Infinity
    lchild_value = self.H[lchild_index] if lchild_index < len(self.H) else float('-inf')
    # set up the value of right child to the element at that index if valid, or else make it +Infinity
    rchild_value = self.H[rchild_index] if rchild_index < len(self.H) else float('-inf')
    # If the value at the index is lessthan or equal to the minimum of two children, then nothing else to do
    if self.H[index] >= max(lchild_value, rchild_value):
        return 
    # Otherwise, find the index and value of the smaller of the two children.
    # A useful python trick is to compare 
    max_child_value, max_child_index = max ((lchild_value, lchild_index), (rchild_value, rchild_index))
    # Swap the current index with the least of its two children
    self.H[index], self.H[max_child_index] = self.H[max_child_index], self.H[index]
    # Bubble down on the minimum child index
    self.bubble_down(max_child_index)
```

##### **Insert Method**
-------------------------------


```python
# Function: insert
# Insert elt into minheap
# Use bubble_up/bubble_down function
def insert(self, elt):
    # your code here
    index = len(self.H)
    self.H.append(elt)
    if index == 1:
        return
    parent_index = index // 2
    parent_index_2 = (index-1) // 2
    if(parent_index == parent_index_2 and self.H[parent_index] < min(self.H[index], self.H[index-1])):
        self.bubble_down(parent_index)
    elif self.H[parent_index] < self.H[index]:
        self.bubble_up(index)
    else:
        return
```

##### **Delete Max.**
-------------------------------


```python
# Function: heap_delete_min
# delete the smallest element in the heap. Use bubble_up/bubble_down
# Function: delete_max
# delete the largest element in the heap. Use bubble_up/bubble_down
def delete_max(self):
    # your code here
    last_index = self.size()
    self.H[1], self.H[last_index] = self.H[last_index], self.H[1]
    self.H.pop(last_index)
    if self.size() > 1:
        self.bubble_down(1)
```


```python
MaxHeap.bubble_up = bubble_up
MaxHeap.bubble_down = bubble_down
MaxHeap.insert = insert
MaxHeap.delete_max = delete_max
```

# <a class="anchor" id="2"><div style="padding:28px;color:white;margin:0;font-size:36px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:48px;background:linear-gradient(100deg, navy, #251cab, #3eb4f4, #251cab)"><span style="color:lightgray">Business Case:</span> The State of Heap Fragmentation</div></a>

### **Demonstration: Create a list of encodings, from MNIST**


```python
from sklearn.datasets import load_digits
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

digits = load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False
)

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

print(predicted.shape)

predicted = predicted.reshape(54, 10)
for prediction in predicted:
    print("-".join(prediction.astype(str).tolist()))
```

    (540,)
    4-0-5-3-6-9-6-1-7-5
    4-4-7-2-8-2-2-5-7-9
    5-4-8-8-4-9-0-8-9-8
    0-1-2-3-4-5-6-7-8-9
    0-1-2-3-4-5-6-7-8-9
    0-1-2-3-4-5-6-7-8-9
    0-9-5-5-6-5-0-9-8-9
    8-4-1-7-7-3-5-1-0-0
    2-2-7-8-2-0-1-2-6-3
    3-7-3-3-4-6-6-6-4-9
    1-5-0-9-6-2-8-3-0-0
    1-7-6-3-2-1-7-4-6-3
    1-3-9-1-7-6-8-4-3-1
    4-0-5-3-6-9-6-1-7-5
    4-4-7-2-8-2-2-5-7-9
    5-4-8-8-4-9-0-8-0-1
    2-3-4-5-6-7-8-9-0-1
    2-3-4-5-6-7-8-9-0-1
    2-3-4-5-6-7-8-9-0-9
    5-5-6-5-0-9-8-9-8-4
    1-7-7-3-5-1-0-0-2-2
    7-8-2-0-1-2-6-3-3-7
    3-3-4-6-6-6-4-9-1-5
    0-9-5-2-8-2-0-0-1-7
    6-3-2-1-7-4-6-3-1-3
    9-1-7-6-8-4-3-1-4-0
    5-3-6-9-6-1-7-5-4-4
    7-2-8-2-2-5-7-9-5-4
    8-8-4-9-0-8-9-8-0-1
    2-3-4-5-1-7-1-9-0-1
    2-3-4-5-6-9-0-1-2-3
    4-5-6-7-8-9-4-9-5-5
    6-5-0-9-8-9-8-4-1-7
    7-3-5-1-0-0-2-2-7-8
    2-0-1-2-6-8-3-7-3-3
    4-6-6-6-9-9-1-5-0-9
    5-2-8-0-1-7-6-3-2-1
    7-9-6-3-1-3-9-1-7-6
    8-4-3-1-4-0-5-3-6-9
    6-1-7-5-4-4-7-2-2-5
    7-3-5-8-4-5-0-8-9-8
    0-1-2-3-4-5-6-7-8-9
    0-1-2-3-4-5-6-7-8-9
    0-1-2-8-4-5-6-7-8-9
    0-9-5-5-6-5-0-9-8-9
    8-4-1-7-7-3-5-1-0-0
    2-2-7-8-2-0-1-2-6-8
    8-7-5-3-4-6-6-6-4-9
    1-5-0-9-5-2-8-2-0-0
    1-7-6-3-2-1-7-4-6-3
    1-3-9-1-7-6-8-4-5-1
    4-0-5-3-6-9-6-1-7-5
    4-4-7-2-8-2-2-5-7-9
    5-4-8-8-4-9-0-8-9-8


### **A List of Encodings, in Max-Heap**


```python
from tqdm import tqdm

list_of_heap_arrays = []

for prediction in tqdm(predicted):
    # for each prediction create a MaxHeap Data Structure
    heap_array = MaxHeap()
    for val in prediction:
        heap_array.insert(val)
    
    list_of_heap_arrays.append(heap_array)
```

    100%|██████████| 54/54 [00:00<00:00, 29848.76it/s]



```python
list_of_heap_arrays
```




    [[9, 7, 6, 5, 5, 4, 6, 0, 1, 3],
     [9, 8, 4, 7, 7, 2, 2, 2, 5, 4],
     [9, 9, 8, 8, 8, 5, 0, 4, 8, 4],
     [9, 8, 5, 6, 7, 1, 4, 0, 3, 2],
     [9, 8, 5, 6, 7, 1, 4, 0, 3, 2],
     [9, 8, 5, 6, 7, 1, 4, 0, 3, 2],
     [9, 9, 5, 8, 9, 5, 0, 0, 6, 5],
     [8, 7, 5, 4, 7, 1, 3, 1, 0, 0],
     [8, 7, 2, 6, 3, 0, 1, 2, 2, 2],
     [9, 7, 6, 4, 6, 3, 6, 3, 4, 3],
     [9, 6, 8, 3, 5, 0, 2, 1, 0, 0],
     [7, 6, 7, 4, 3, 1, 6, 1, 3, 2],
     [9, 7, 8, 4, 1, 3, 6, 1, 3, 1],
     [9, 7, 6, 5, 5, 4, 6, 0, 1, 3],
     [9, 8, 4, 7, 7, 2, 2, 2, 5, 4],
     [9, 8, 8, 8, 4, 5, 0, 4, 0, 1],
     [9, 8, 7, 5, 4, 3, 6, 2, 0, 1],
     [9, 8, 7, 5, 4, 3, 6, 2, 0, 1],
     [9, 9, 7, 5, 8, 3, 6, 2, 0, 4],
     [9, 9, 8, 8, 4, 5, 6, 5, 5, 0],
     [7, 5, 7, 2, 3, 1, 0, 0, 1, 2],
     [8, 7, 6, 3, 7, 2, 2, 0, 3, 1],
     [9, 6, 6, 6, 5, 3, 4, 3, 1, 4],
     [9, 8, 5, 1, 7, 2, 0, 0, 0, 2],
     [7, 6, 6, 3, 3, 2, 4, 1, 1, 3],
     [9, 8, 7, 4, 6, 4, 3, 1, 1, 0],
     [9, 6, 7, 5, 6, 1, 5, 3, 4, 4],
     [9, 8, 7, 5, 4, 5, 7, 2, 2, 2],
     [9, 8, 9, 8, 1, 4, 8, 8, 0, 0],
     [9, 7, 5, 4, 1, 3, 1, 2, 0, 1],
     [9, 5, 6, 2, 4, 3, 0, 1, 2, 3],
     [9, 9, 8, 7, 6, 5, 4, 4, 5, 5],
     [9, 8, 9, 5, 7, 0, 8, 4, 1, 6],
     [8, 7, 5, 3, 7, 0, 2, 1, 2, 0],
     [8, 7, 6, 3, 3, 1, 3, 0, 2, 2],
     [9, 9, 9, 5, 6, 6, 1, 4, 0, 6],
     [8, 3, 7, 2, 1, 5, 6, 0, 2, 1],
     [9, 7, 9, 7, 6, 3, 6, 1, 3, 1],
     [9, 8, 5, 4, 6, 0, 3, 1, 3, 4],
     [7, 5, 7, 2, 5, 4, 6, 1, 2, 4],
     [9, 8, 5, 8, 8, 5, 0, 3, 7, 4],
     [9, 8, 5, 6, 7, 1, 4, 0, 3, 2],
     [9, 8, 5, 6, 7, 1, 4, 0, 3, 2],
     [9, 8, 6, 7, 8, 1, 5, 0, 4, 2],
     [9, 9, 5, 8, 9, 5, 0, 0, 6, 5],
     [8, 7, 5, 4, 7, 1, 3, 1, 0, 0],
     [8, 8, 2, 6, 7, 0, 1, 2, 2, 2],
     [9, 8, 6, 6, 7, 5, 6, 3, 4, 4],
     [9, 5, 8, 2, 5, 0, 2, 1, 0, 0],
     [7, 6, 7, 4, 3, 1, 6, 1, 3, 2],
     [9, 7, 8, 5, 1, 3, 6, 1, 4, 1],
     [9, 7, 6, 5, 5, 4, 6, 0, 1, 3],
     [9, 8, 4, 7, 7, 2, 2, 2, 5, 4],
     [9, 9, 8, 8, 8, 5, 0, 4, 8, 4]]




```python
encodings = [[9, 7, 6, 5, 5, 4, 6, 0, 1, 3],
[9, 8, 4, 7, 7, 2, 2, 2, 5, 4],
[9, 9, 8, 8, 8, 5, 0, 4, 8, 4]]

# tree1 :: Tree Int
tree1 = Node(9)([
    Node(7)([
        Node(5)([
            Node(0)([]),
            Node(1)([])
        ]),
        Node(5)([
            Node(3)([])
        ])
    ]),
    Node(6)([
        Node(4)([]),
        Node(6)([])
    ])
])

tree2 = Node(9)([
    Node(8)([
        Node(7)([
            Node(2)([]),
            Node(5)([])
        ]),
        Node(7)([
            Node(4)([])
        ])
    ]),
    Node(4)([
        Node(2)([]),
        Node(2)([])
    ])
])

tree3 = Node(9)([
    Node(9)([
        Node(8)([
            Node(4)([]),
            Node(8)([])
        ]),
        Node(8)([
            Node(4)([])
        ])
    ]),
    Node(8)([
        Node(5)([]),
        Node(0)([])
    ])
])


print('\n\n'.join([
    'Centered parents with nodeless lines pruned out:',
    drawTree2(False)(True)(
        tree1
    ),
    
    drawTree2(False)(True)(
        tree2
    ),
    
    drawTree2(False)(True)(
        tree3
    )
]))
```

    Centered parents with nodeless lines pruned out:
    
               ┌ 0 
           ┌ 5 ┤
       ┌ 7 ┤   └ 1 
     9 ┤   └ 5 ─ 3 
       │   ┌ 4 
       └ 6 ┤
           └ 6 
    
               ┌ 2 
           ┌ 7 ┤
       ┌ 8 ┤   └ 5 
     9 ┤   └ 7 ─ 4 
       │   ┌ 2 
       └ 4 ┤
           └ 2 
    
               ┌ 4 
           ┌ 8 ┤
       ┌ 9 ┤   └ 8 
     9 ┤   └ 8 ─ 4 
       │   ┌ 5 
       └ 8 ┤
           └ 0 


# <a class="anchor" id="2"><div style="padding:28px;color:white;margin:0;font-size:36px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:48px;background:linear-gradient(100deg, navy, #251cab, #3eb4f4, #251cab)"><span style="color:lightgray">Business Case:</span> An Example of Traceability through Number Plate Recognition</div></a>

# <a class="anchor" id="2"><div style="padding:20px;color:white;margin:0;font-size:80%;text-align:left;display:fill;border-radius:10px;background-color:#251cab;overflow:hidden;background:linear-gradient(100deg, maroon, brown, darkorange, darkorange)">Research Question 2 (RQ2):</div></a>
-----------------------------------

### **How will the Networking Frame be represented, such that it contains some Traceability Information from a single page of text?**

## Import Libraries


```python
import fcntl
import socket
import struct
import sys
```

# <a class="anchor" id="2"><div style="padding:28px;color:black;margin:0;font-size:36px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:48px;background:linear-gradient(100deg, aqua, aqua, aqua, aqua)"><span style="color:darkgray">Use Case:</span> Numberplate Recognition</div></a>

**Reference:**

##### **RoboFlow: Number Plate Identification using numbers-identification Detection API**

https://universe.roboflow.com/yashwanthworkspace/numbers-identification/model/1


```python
display.Image("./inference_images/Cars147.png")
```




    
![png](output_86_0.png)
    




```python
display.Image("./inference_images/Cars147_crop.png", width=500)
```




    
![png](output_87_0.png)
    



### **Number Plate Recognition**


```python
API_KEY = "ByW1jMeLjvRKbANdWHCJ"

from roboflow import Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("numbers-identification")
model = project.version(1).model

# infer on a local image
print(model.predict("inference_images/Cars147_crop.png", confidence=5, overlap=40).json())

# visualize your prediction
model.predict("inference_images/Cars147_crop.png", confidence=5, overlap=40).save("results_image/prediction.jpg")
```

    loading Roboflow workspace...
    loading Roboflow project...
    {'predictions': [{'x': 124.0, 'y': 51.0, 'width': 18.0, 'height': 36.0, 'confidence': 0.16010162234306335, 'class': 'B', 'image_path': 'inference_images/Cars147_crop.png', 'prediction_type': 'ObjectDetectionModel'}, {'x': 155.0, 'y': 48.0, 'width': 18.0, 'height': 34.0, 'confidence': 0.12469622492790222, 'class': '-2-', 'image_path': 'inference_images/Cars147_crop.png', 'prediction_type': 'ObjectDetectionModel'}, {'x': 53.5, 'y': 55.5, 'width': 19.0, 'height': 39.0, 'confidence': 0.10968968272209167, 'class': 'K', 'image_path': 'inference_images/Cars147_crop.png', 'prediction_type': 'ObjectDetectionModel'}, {'x': 186.5, 'y': 48.0, 'width': 17.0, 'height': 32.0, 'confidence': 0.05467239022254944, 'class': '-5-', 'image_path': 'inference_images/Cars147_crop.png', 'prediction_type': 'ObjectDetectionModel'}], 'image': {'width': '235', 'height': '90'}}



```python
display.Image("results_image/prediction.jpg", width=500)
```




    
![jpeg](output_90_0.jpg)
    



### **Machine Learning Validation**


```python
predictions = model.predict("inference_images/Cars147_crop.png", confidence=5, overlap=40).json()
```


```python
import numpy as np

recog = {}
stencil = [[51, 71], [69, 89], [86, 106], [102, 122], [120, 144], [152, 173], [185, 205]]
plate = ['K', 'L', '0', '1', 'C', 'A', '2', '5', '5', '5']
idx = 0
for pred in predictions['predictions']:
    x, _, width, _, conf, det, image_path, pred_type = pred.values()
    for sten in stencil:
        if np.isclose(sten[0], x, atol=3.0) and np.isclose(sten[1], x+width, atol=3.0):
            recog[tuple(sten)] = det
            idx += 1
```

### **In-Sample and Out-of-Sample Error**


```python
error = [1, 1, 1, 1, 1, 1, 1]
idx = 0
for key, value in recog.items():
    try:
        idx += 1
        if plate.index(value.strip("![a-zA-Z0-9]")) >= 0:
            error[plate.index(value.strip("![a-zA-Z0-9]"))] = 0
    except Exception as e:
        continue
```


```python
print("""In-Sample error for 1 sample: {0} out of {1}""".format(idx - (len(error) - sum(error)), idx))
print("""Out-of-Sample error for 1 sample: {0} out of {1}""".format(sum(error), len(error) - idx))
```

    In-Sample error for 1 sample: 0 out of 2
    Out-of-Sample error for 1 sample: 5 out of 5


### **Identification of Number Plate**

* Indian Number Plate Recognition Format


```python
display.Image("./images/IND-license-plate-en.jpg")
```




    
![jpeg](output_98_0.jpg)
    



#### **Attributes of the Number Plate**


```python
STATE = 'KL'
REGIONAL_TRANSPORT_AUTHORITY = '01'
REG_SERIES = 'CA'
UNIQUE_NUMBER = '2555'
```

#### **5 Levels of Identifiers**


```python
display.Image("./images/vexf-structure.png")
```




    
![png](output_102_0.png)
    



### **A Networking Wi-Fi Frame**

**Reference:**

https://gist.github.com/iximiuz/fe49427ac6900f2a590629500981e5ab

### **THE BELOW IS A LOCKED CELL**


```python
!rm -rf /tmp/test_frame.sock

import sys
import fcntl
import struct
import socket as s
import termios
import array, fcntl, struct, termios, os
import sys

# How to send a packet
# (ifname, proto[, pkttype[, hatype[, addr]]])

def human_mac_to_bytes(addr):
    return bytes.fromhex(addr.replace(':', ''))

ifname='eth0'
dstmac='3D:91:57:CA:82:03'
payload='1234567890'
eth_type=b'\x7A\x05'

def create_frame(ifname, dstmac, eth_type, payload):
    
    print(os.getpgrp())
    info = fcntl.ioctl(1, termios.TIOCGPGRP, struct.pack('256s', bytes('wlo1', 'utf-8')[:15]), 1)
    srcmac = ':'.join('%02x' % b for b in info[18:24])

    # Build Ethernet frame
    payload_bytes = payload.encode('utf-8')
    assert len(payload_bytes) <= 1500  # Ethernet MTU

    frame = human_mac_to_bytes(dstmac) + \
            human_mac_to_bytes(srcmac) + \
            eth_type + \
            payload_bytes

    # Create Ethernet frame
    return frame
```

### **An ioctl request made to the device**

* os.O_RDONLY
* os.O_WRONLY
* os.O_RDWR
* os.O_APPEND
* os.O_CREAT
* os.O_EXCL
* os.O_TRUNC


```python
import ctypes
import os
import ioctl
import ioctl.linux

def make_ioctl_request():
    RNDGETENTCNT = ioctl.linux.IOR('R', 0x00, ctypes.c_int)
    rndgetentcnt = ioctl.ioctl_fn_ptr_r(RNDGETENTCNT, ctypes.c_int)

    fd = os.open('/dev/random', os.O_RDONLY)
    entropy_avail = rndgetentcnt(fd)
    print('entropy_avail:', entropy_avail)
    
make_ioctl_request()
```

    entropy_avail: 3571


#### **Visualize a Frame**


```python
display.Image("./images/code_involved_creating_frame.png")
```




    
![png](output_108_0.png)
    




```python
display.Image("./images/create_a_frame.png")
```




    
![png](output_109_0.png)
    



# <a class="anchor" id="2"><div style="padding:28px;color:white;margin:0;font-size:36px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:48px;background:linear-gradient(100deg, navy, #251cab, #3eb4f4, #251cab)"><span style="color:lightgray">Business Case:</span> The Vision Metadata used for Discoverability and Private & Secure Data Exchange</div></a>

# <a class="anchor" id="2"><div style="padding:20px;color:white;margin:0;font-size:80%;text-align:left;display:fill;border-radius:10px;background-color:#251cab;overflow:hidden;background:linear-gradient(100deg, maroon, brown, darkorange, darkorange)">Research Question 3 (RQ3):</div></a>
-----------------------------------

### **An HTTP Response for Discoverability of Object Detection Bounding Boxes**


```python
import requests
from pprint import PrettyPrinter
pp = PrettyPrinter()
```

**Reference:**

##### **11.3.  New Header Fields**
https://datatracker.ietf.org/doc/html/rfc3903#section-11.3

##### ****11.1.1.  PUBLISH Method****

"PUBLISH" is added to the definition of the element "Method" in the
SIP message grammar.  As with all other SIP methods, the method name
is case sensitive.  PUBLISH is used to publish event state to an
entity responsible for compositing this event state.

https://www.pinsentmasons.com/out-law/news/electronic-product-codes-and-rfid-privacy-guidance

# <a class="anchor" id="2"><div style="padding:28px;color:black;margin:0;font-size:36px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:48px;background:linear-gradient(100deg, aqua, aqua, aqua, aqua)"><span style="color:darkgray">Use Case:</span> Privacy-Preserved Object Recognition</div></a>

### **From a Sample Image, transmit data through If-Match and If-None-Match**


```python
import boto3

client = boto3.client(service_name='s3', aws_access_key_id='AKIATHZWSE5UPJVXFRR3', aws_secret_access_key='3SbOWy4Sk2iG1zLopfP+sIrUlc1RuF3wNWz5kBr6')
```


```python
client.generate_presigned_url(
    "get_object", ExpiresIn=1000000, Params={"Bucket": 'virtual-demand-inspection', "Key": 'intel-oneapi-hackathon/construction-dataset/1-bank-western-view-construction-dataset.jpg'}
)
```




    'https://virtual-demand-inspection.s3.eu-west-2.amazonaws.com/intel-oneapi-hackathon/construction-dataset/1-bank-western-view-construction-dataset.jpg?AWSAccessKeyId=AKIATHZWSE5UPJVXFRR3&Signature=yReCf%2FSPXqpDREbRBQFIK%2Bal5ks%3D&Expires=1678582390'




```python
display.Image("https://virtual-demand-inspection.s3.eu-west-2.amazonaws.com/intel-oneapi-hackathon/construction-dataset/1-bank-western-view-construction-dataset.jpg")
```




    
![jpeg](output_118_0.jpg)
    



### **Request RoboFlow API from a Trained Model on Construction Dataset**


```python
from roboflow import Roboflow
rf = Roboflow(api_key="ByW1jMeLjvRKbANdWHCJ")
project = rf.workspace().project("construction-dataset-6xih3")
model = project.version(2).model

# infer on a local image
model.predict("inference_images/2-construction-site-water-flooded.jpg", confidence=10, overlap=40).save('results_image/construction-dataset-machinery-detected-bbox.jpg')
```

    loading Roboflow workspace...
    loading Roboflow project...



```python
from roboflow import Roboflow
rf = Roboflow(api_key="ByW1jMeLjvRKbANdWHCJ")
project = rf.workspace().project("construction-dataset-6xih3")
model = project.version(2).model

# infer on a local image
model.predict("inference_images/1-bank-western-view-construction-dataset-resized.jpg", confidence=10, overlap=40).save('results_image/construction-dataset-buildings-detected-bbox.jpg')
```

    loading Roboflow workspace...
    loading Roboflow project...


### **HTTP Response with the E-tag returned on Sending a HTTP Request**


```python
display.Image("results_image/construction-dataset-buildings-detected-bbox.jpg")
```




    
![jpeg](output_123_0.jpg)
    



### E-Tag as Bounding Boxes


```python
display.Image("results_image/construction-dataset-machinery-detected-bbox.jpg")
```




    
![jpeg](output_125_0.jpg)
    




```python
from roboflow import Roboflow
rf = Roboflow(api_key="ByW1jMeLjvRKbANdWHCJ")
project = rf.workspace().project("construction-dataset-6xih3")
model = project.version(2).model

# infer on a local image
model.predict("inference_images/2-construction-site-water-flooded.jpg", confidence=89, overlap=40).save('results_image/construction-dataset-machinery-detected-bbox-observability-by-parts.jpg')
```

    loading Roboflow workspace...
    loading Roboflow project...


# <a class="anchor" id="2"><div style="padding:28px;color:white;margin:0;font-size:36px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:48px;background:linear-gradient(100deg, darkgreen, darkgreen, darkgreen, darkgreen)"><span style="color:lightgray">Business Case:</span> Observability By Parts, Integrating Safety at Sites</div></a>


```python
display.Image("results_image/construction-dataset-machinery-detected-bbox-observability-by-parts.jpg")
```




    
![jpeg](output_128_0.jpg)
    




```python
display.Image("./images/object-detection-prediction-bounding-boxes.JPG")
```




    
![jpeg](output_129_0.jpg)
    



### **Visualize HTTP Response**


```python
response = client.get_object(
    Bucket='virtual-demand-inspection',
    Key='intel-oneapi-hackathon/construction-dataset/1-bank-western-view-construction-dataset.jpg',
    IfMatch='d0786438ef5332eedbd55c38e61cf19a'
)

pp.pprint(response)
```

    {'AcceptRanges': 'bytes',
     'Body': <botocore.response.StreamingBody object at 0x7f79ac9f4940>,
     'ContentLength': 2878587,
     'ContentType': 'image/jpeg',
     'ETag': '"d0786438ef5332eedbd55c38e61cf19a"',
     'LastModified': datetime.datetime(2023, 2, 28, 10, 10, 11, tzinfo=tzutc()),
     'Metadata': {},
     'ResponseMetadata': {'HTTPHeaders': {'accept-ranges': 'bytes',
                                          'content-length': '2878587',
                                          'content-type': 'image/jpeg',
                                          'date': 'Wed, 01 Mar 2023 21:50:35 GMT',
                                          'etag': '"d0786438ef5332eedbd55c38e61cf19a"',
                                          'last-modified': 'Tue, 28 Feb 2023 '
                                                           '10:10:11 GMT',
                                          'server': 'AmazonS3',
                                          'x-amz-id-2': 'rrN/Qhy5TXClOq+A6yXVOGCleQzHwSmiweWeadKcIJf/p/yTOUR5S3qsrbm4r9GMKki9ho6dWXw=',
                                          'x-amz-request-id': 'B4SGF6V9MHWG4VZN',
                                          'x-amz-server-side-encryption': 'AES256',
                                          'x-amz-tagging-count': '1'},
                          'HTTPStatusCode': 200,
                          'HostId': 'rrN/Qhy5TXClOq+A6yXVOGCleQzHwSmiweWeadKcIJf/p/yTOUR5S3qsrbm4r9GMKki9ho6dWXw=',
                          'RequestId': 'B4SGF6V9MHWG4VZN',
                          'RetryAttempts': 1},
     'ServerSideEncryption': 'AES256',
     'TagCount': 1}


# <a class="anchor" id="2"><div style="padding:28px;color:white;margin:0;font-size:24px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:36px;background:linear-gradient(100deg, darkpurple, darkpurple, darkpurple, darkpurple)"><span style="color:lightgray">Business Case:</span> Turn off False Alarms: An ioctl Request to Send Control Codes to the Device Drivers</div></a>

**Reference:**

https://github.com/aswinvk28/python-ioctl/


```python
import sys
import fcntl
import struct
import socket as s
import termios
import array, fcntl, struct, termios, os
import sys

from scapy.all import *
```


```python
def human_mac_to_bytes(addr):
    return bytes.fromhex(addr.replace(':', ''))

def create_frame(ifname, dstmac, eth_type, payload):
    
    print(os.getpgrp())
    info = fcntl.ioctl(0, termios.TIOCGPGRP, struct.pack('256s', bytes('wlo1', 'utf-8')[:15]), 1)
    srcmac = ':'.join('%02x' % b for b in info[18:24])

    # Build Ethernet frame
    payload_bytes = payload.encode('utf-8')
    assert len(payload_bytes) <= 1500  # Ethernet MTU

    frame = human_mac_to_bytes(dstmac) + \
            human_mac_to_bytes(srcmac) + \
            eth_type + \
            payload_bytes

    # Create Ethernet frame
    return frame

def create_packet(dstaddress, udp_port):
    return IP(dst=dstaddress)/UDP(dport=udp_port)
```

### Protocol 'HANDSHAKE' to register the device whenever a change is detected


```python
# 'REQUEST' to create VEXF Frame with IoT Device details: with Privacy Protection
ifname='wlo1'
dstmac='3D:91:57:CA:82:03'
payload='Generating VEXF Handshake Frame to link the IoT device to a Request Broker using UDP'
eth_type=b'HANDSHAKE'
```


```python
create_frame(ifname, dstmac, eth_type, payload)
```

### Protocol 'REQUEST' to create ioctl 'IOR' to talk to another driver from the WiFi Driver


```python
# 'REQUEST' to create VEXF Frame with IoT Device details: with Privacy Protection
ifname='wlo1'
dstmac='3D:91:57:CA:82:03'
payload='Request to create an ioctl control code for initiating a hand-shake'
eth_type=b'REQUEST'
```


```python
create_frame(ifname, dstmac, eth_type, payload)
```

### Protocol 'VEXFRESPONSE' to provide a HTTP Response

* Scapy

### THE BELOW IS A LOCKED CELL


```python
ip = create_packet("8.8.8.8", 80)
payload=b"VEXF Frame Payload in application/octet-stream"
data_packet = TCP(sport=1500, dport=80, flags="", seq=102)    # Notice that we explicitly mentioned to not set any flags

send(ip/data_packet/payload)
```

## **Private and Secure Data Exchange using VEXF Protocol**


```python
display.Image("./images/VEXF - Sequence Diagram.png")
```




    
![png](output_144_0.png)
    



# <a class="anchor" id="2"><div style="padding:28px;color:white;margin:0;font-size:24px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:36px;background:linear-gradient(100deg, darkpurple, darkpurple, darkpurple, darkpurple)"><span style="color:lightgray">Business Case:</span> VEXF Within eBPF Linux Kernel</div></a>

# <a class="anchor" id="2"><div style="padding:20px;color:white;margin:0;font-size:80%;text-align:left;display:fill;border-radius:10px;background-color:#251cab;overflow:hidden;background:linear-gradient(100deg, maroon, brown, darkorange, darkorange)">Research Question 4 (RQ4):</div></a>
-----------------------------------

### **Validate a VEXF Struct using eBPF in a JIT Compilation**

* Code Sample

### THE BELOW IS A LOCKED CELL

# <a class="anchor" id="2"><div style="padding:28px;color:black;margin:0;font-size:36px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:48px;background:linear-gradient(100deg, aqua, aqua, aqua, aqua)"><span style="color:darkgray">Use Case:</span> Using eBPF for Observability</div></a>


```python
%%writefile executable/bpcc_ebpf_hey.py

from bcc import BPF

program = """
BPF_PERF_OUTPUT(sugar);
struct data_t {
    u32 pid;
    char command[16];
    char message[12];
};
int hello(void *ctx) {
    struct data_t data = {};
    char message[12] = "Hello World";
    data.pid = bpf_get_current_pid_tgid();
    bpf_get_current_comm(&data.command, sizeof(data.command));
    bpf_probe_read_kernel(&data.message, sizeof(data.message), message);
    sugar.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

b = BPF(text=program)

print(b)
```

    Overwriting executable/bpcc_ebpf_hey.py



```python
!cd executable/ && sudo --prompt="ebpfuser" python3 ./bpcc_ebpf_hey.py
```

    <bcc.BPF object at 0x7f7f1a2d3c10>


### **Design a VEXF Struct inside Kernel Space**


```python
"""
BPF_PERF_OUTPUT(sugar);
struct data_t {
    u32 pid;
    char command[16];
    char message[12];
};
int hello(void *ctx) {
    struct data_t data = {};
    char message[12] = "Hello World";
    data.pid = bpf_get_current_pid_tgid();
    bpf_get_current_comm(&data.command, sizeof(data.command));
    bpf_probe_read_kernel(&data.message, sizeof(data.message), message);
    sugar.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""
```




    '\nBPF_PERF_OUTPUT(sugar);\nstruct data_t {\n    u32 pid;\n    char command[16];\n    char message[12];\n};\nint hello(void *ctx) {\n    struct data_t data = {};\n    char message[12] = "Hello World";\n    data.pid = bpf_get_current_pid_tgid();\n    bpf_get_current_comm(&data.command, sizeof(data.command));\n    bpf_probe_read_kernel(&data.message, sizeof(data.message), message);\n    sugar.perf_submit(ctx, &data, sizeof(data));\n    return 0;\n}\n'



### **Write a User-Space and Kernel-Space eBPF Program to be sent to networking interfaces**


```python
%%writefile executable/bpcc_ebpf_sugar.py

#!/usr/bin/python
from bcc import BPF

program = """
BPF_PERF_OUTPUT(sugar);
struct data_t {
    u32 pid;
    char command[16];
    char message[12];
};
int hello(void *ctx) {
    struct data_t data = {};
    char message[12] = "Hello World";
    data.pid = bpf_get_current_pid_tgid();
    bpf_get_current_comm(&data.command, sizeof(data.command));
    bpf_probe_read_kernel(&data.message, sizeof(data.message), message);
    sugar.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

b = BPF(text=program)
syscall = b.get_syscall_fnname("execve")
b.attach_kprobe(event=syscall, fn_name="hello")

def print_event(cpu, data, size):
    data = b["sugar"].event(data)
    print("{0} {1} {2}".format(data.pid, data.command.decode(), data.message.decode()))

b["sugar"].open_perf_buffer(print_event)

while True:
    b.perf_buffer_poll()
```

    Overwriting executable/bpcc_ebpf_sugar.py



```python
!cd executable/ && sudo --prompt="ebpfuser" python3 bpcc_ebpf_sugar.py -v
```

    18277 colord Hello World
    18287 (spawn) Hello World
    18288 colord Hello World
    18293 (spawn) Hello World
    18295 (spawn) Hello World
    18296 (spawn) Hello World
    ^C
    Traceback (most recent call last):
      File "/opt/ebpfuser/image-recognition-with-privacy/executable/bpcc_ebpf_sugar.py", line 34, in <module>
        b.perf_buffer_poll()
      File "/usr/lib/python3/dist-packages/bcc/__init__.py", line 1677, in perf_buffer_poll
        lib.perf_reader_poll(len(readers), readers, timeout)
    KeyboardInterrupt


### **Output**


```python
display.Image("images/eBPF-Demo-Code.png")
```




    
![png](output_156_0.png)
    



### **Visualization on Objects involved in Privacy-Enhancement**


```python
display.Image("images/construction-dataset-machinery-detected.png")
```




    
![png](output_158_0.png)
    



# <a class="anchor" id="2"><div style="padding:28px;color:white;margin:0;font-size:24px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:36px;background:linear-gradient(100deg, darkpurple, darkpurple, darkpurple, darkpurple)"><span style="color:lightgray">Business Case:</span> Detecting Interference in Laser within an Interferometer</div></a>


```python
display.Image("./images/vexf-packet-interferometer.JPG")
```




    
![jpeg](output_160_0.jpg)
    




```python
from IPython.display import YouTubeVideo

YouTubeVideo('UA1qG7Fjc2A', width=800, height=300)
```





<iframe
    width="800"
    height="300"
    src="https://www.youtube.com/embed/UA1qG7Fjc2A"
    frameborder="0"
    allowfullscreen

></iframe>





```python
display.Image("./images/vexf-structure-new.png")
```




    
![png](output_162_0.png)
    



### **The Heap-Structure Encoding**


```python
display.Image("./images/heap.structure.png", width=600)
```




    
![png](output_164_0.png)
    



# <a class="anchor" id="2" style="color:black"><div style="padding:28px;color:white;margin:0;font-size:24px;font-weight:bold;text-align:center;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:36px;background:linear-gradient(100deg, darkgray, darkgray, darkgray, darkgray)"><span style="color:lightgray">Feature Engineering:</span> Text Recognition of Synthetic Dataset</div></a>

### Download the PaddleOCR Pre-Trained Model


```python
!curl https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar -o  models/en_PP-OCRv3_rec_infer.tar
```

### Convert PaddleOCR Model to ONNX Model


```python
!paddle2onnx --model_dir models/en_PP-OCRv3_rec_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file models/paddleocr.onnx \
--enable_dev_version True
```

## The Dataset


```python
img_dir_path = "./projects/TextRecognitionDataGenerator/out"
img_file_path = "*.jpg"

sample_dataset = glob(os.path.join(img_dir_path, img_file_path))
print("Length of the Dataset: ", len(sample_dataset))
```

    Length of the Dataset:  1000


## The Dataset by Clusters on Books with Alphabets


```python
from pprint import PrettyPrinter

img_dir_path = "./projects/TextRecognitionDataGenerator/out"
alphabets = 'abcdefghijklmnopqrstuvwxyz'

pp = PrettyPrinter()

for letter in alphabets:
    img_file_path = letter + "*.jpg"
    sample_dataset = glob(os.path.join(img_dir_path, img_file_path))
    clusters[letter] = len(sample_dataset)

pp.pprint(clusters)

print("Total number of text images starting with a-z: ", sum(clusters.values()), " text images")
```

    {'a': 51,
     'b': 31,
     'c': 74,
     'd': 38,
     'e': 26,
     'f': 29,
     'g': 18,
     'h': 36,
     'i': 33,
     'j': 5,
     'k': 4,
     'l': 23,
     'm': 38,
     'n': 32,
     'o': 32,
     'p': 80,
     'q': 10,
     'r': 34,
     's': 92,
     't': 36,
     'u': 43,
     'v': 23,
     'w': 26,
     'x': 1,
     'y': 3,
     'z': 1}
    Total number of text images starting with a-z:  819  text images


## Sample Data from the Text Recognition Dataset

**Reference:**

https://github.com/Belval/TextRecognitionDataGenerator


```python
from glob import glob
import numpy as np
import os

img_dir_path = "./projects/TextRecognitionDataGenerator/out"
img_file_path = "*.jpg"
sample_size = 20
sample_dataset = np.random.choice(glob(os.path.join(img_dir_path, img_file_path)), size=sample_size)
```


```python
import matplotlib.pyplot as plt
import cv2

fig, ax = plt.subplots(sample_size//2, 2, figsize=(12,12))

first_letters = []
for data_strip in sample_dataset:
    first_letters.append(os.path.splitext(data_strip)[0].split("/")[-1][0])

for ii, data_strip in enumerate(sample_dataset):
    ax[ii//2, ii%2].set_title("Cluster " + first_letters[ii])
    img = cv2.imread(data_strip, cv2.IMREAD_UNCHANGED)
    ax[ii//2, ii%2].imshow(img)
fig.tight_layout(pad=2.0)
```


    
![png](output_176_0.png)
    


## Inference from Pre-Trained Downloaded Model using OpenVINO

#### Read the Model


```python
from openvino.inference_engine import IECore
from openvino.runtime import Core

# Creating Inference Engine
runtime = Core()
device = 'CPU'

onnx_model_path = './models/en_PP-OCRv3_rec_infer/inference.pdmodel'

# Read a model in OpenVINO Intermediate Representation or ONNX format
print("Reading the network:", onnx_model_path)
# (.xml and .bin files) or (.onnx file)
model = runtime.read_model(model=onnx_model_path)
```

    Reading the network: ./models/en_PP-OCRv3_rec_infer/inference.pdmodel


### Model Compiling

#### Compile the Model


```python
compiled_model = runtime.compile_model(model=model, device_name="CPU")

# Get input and output nodes for text detection.
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

print(input_layer, output_layer)
```

    <ConstOutput: names[x] shape[?,3,48,?] type: f32> <ConstOutput: names[softmax_2.tmp_0] shape[?,1..,97] type: f32>


### Print the Model metadata


```python
# check output layers
print(f"output shape: {model.get_output_shape}")
print(f"output size: {model.get_output_size()}")
print(f"result: {model.get_result()}")
```

    output shape: <bound method PyCapsule.get_output_shape of <Model: 'Model0'
    inputs[
    <ConstOutput: names[x] shape[?,3,48,?] type: f32>
    ]
    outputs[
    <ConstOutput: names[softmax_2.tmp_0] shape[?,1..,97] type: f32>
    ]>>
    output size: 1
    result: <Result: 'softmax_2.tmp_0/Result' ([?,1..,97])>


### Inference of Single Image

**Reference:**

EastProcessTrain

https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/ppocr/data/imaug/east_process.py


```python
from tqdm import tqdm

image_shape = [48, 48]
background_ratio = 0.125
min_crop_side_ratio = 0.1
min_text_size = 10

input_size = image_shape[0]

# Preprocess the image
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/ppocr/data/imaug/east_process.py
def preprocess(im):
    x_dim = im.shape[2]
    y_dim = im.shape[1]
    im = cv2.resize(im, (int(y_dim / x_dim * image_shape[0]), image_shape[0]))
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    # im = im[:, :, ::-1].astype(np.float32)
    im = im / 255
    im -= img_mean
    im /= img_std
    new_h, new_w, _ = im.shape
    im_padded = np.zeros((new_h, new_w, 3), dtype=np.float32)
    im_padded[:new_h, :new_w, :] = im
    im_padded = im_padded.transpose((2, 0, 1))
    im_padded = im_padded[np.newaxis, :]
    return im_padded

# Collect Images
img_array = []
for data_strip in tqdm(sample_dataset):
    img_array.append(cv2.imread(data_strip, cv2.IMREAD_UNCHANGED))

# Apply Pre-process Function
for ii, img in tqdm(enumerate(img_array)):
    img_array[ii] = preprocess(img)
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 702.08it/s]
    20it [00:00, 23.23it/s]


### Text Recognition by Bounding Boxes


```python
# View the compiled model
compiled_model([img_array[0]])[output_layer]
```




    array([[[9.4866765e-01, 2.1827147e-04, 6.3265339e-03, ...,
             8.3427833e-05, 1.0560153e-06, 1.0911346e-03],
            [9.6773249e-01, 8.0632058e-04, 3.2007401e-03, ...,
             8.2764593e-05, 1.8393096e-06, 1.0504680e-03],
            [8.3332413e-01, 6.4523821e-03, 7.1721468e-03, ...,
             5.2311143e-04, 1.4662880e-05, 3.0928191e-03],
            ...,
            [9.6507907e-01, 2.8534431e-03, 2.1830075e-03, ...,
             7.5988399e-05, 3.7163827e-06, 7.6215016e-04],
            [9.2042571e-01, 5.3987550e-03, 4.5434008e-03, ...,
             1.6335557e-04, 7.1875957e-06, 1.2620544e-03],
            [9.3937337e-01, 4.7850798e-04, 5.6498097e-03, ...,
             1.7833467e-04, 4.9132273e-06, 3.1985904e-03]]], dtype=float32)



# EasyOCR

* A PyTorch based Model

### A Demo of Inference


```python
import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
result = reader.readtext('./projects/TextRecognitionDataGenerator/out/zircite Szeklian dildoes ICLID macroseism_819.jpg')
```

    CUDA not available - defaulting to CPU. Note: This module is much faster with a GPU.
    Downloading detection model, please wait. This may take several minutes depending upon your network connection.


    Progress: |██████████████████████████████████████████████████| 100.0% Complete

    Downloading recognition model, please wait. This may take several minutes depending upon your network connection.


    Progress: |██████████████████████████████████████████████████| 100.0% Complete


```python
result
```




    [([[0, 7], [1051, 7], [1051, 64], [0, 64]],
      'zircite Szeklian dildoes ICLID macroseism',
      0.8294350822562625)]



## EasyOCR Framework


```python
display.Image("./images/easyocr_framework.jpeg")
```




    
![jpeg](output_191_0.jpg)
    



## OpenVINO Model


```python
!omz_downloader --name "text-spotting-*" --output_dir "./models/"
```

    ################|| Downloading text-spotting-0005-detector ||################
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-detector/FP32/text-spotting-0005-detector.xml
    ... 100%, 670 KB, 1036 KB/s, 0 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-detector/FP32/text-spotting-0005-detector.bin
    ... 100%, 101364 KB, 3963 KB/s, 25 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-detector/FP16/text-spotting-0005-detector.xml
    ... 100%, 777 KB, 1143 KB/s, 0 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-detector/FP16/text-spotting-0005-detector.bin
    ... 100%, 50682 KB, 3919 KB/s, 12 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-detector/FP16-INT8/text-spotting-0005-detector.xml
    ... 100%, 991 KB, 1385 KB/s, 0 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-detector/FP16-INT8/text-spotting-0005-detector.bin
    ... 100%, 25481 KB, 3782 KB/s, 6 seconds passed
    
    ################|| Downloading text-spotting-0005-recognizer-encoder ||################
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-recognizer-encoder/FP32/text-spotting-0005-recognizer-encoder.xml
    ... 100%, 8 KB, 56565 KB/s, 0 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-recognizer-encoder/FP32/text-spotting-0005-recognizer-encoder.bin
    ... 100%, 5187 KB, 2946 KB/s, 1 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-encoder.xml
    ... 100%, 11 KB, 78620 KB/s, 0 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-encoder.bin
    ... 100%, 2593 KB, 2315 KB/s, 1 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16-INT8/text-spotting-0005-recognizer-encoder.xml
    ... 100%, 16 KB, 85339 KB/s, 0 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16-INT8/text-spotting-0005-recognizer-encoder.bin
    ... 100%, 2593 KB, 2301 KB/s, 1 seconds passed
    
    ################|| Downloading text-spotting-0005-recognizer-decoder ||################
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-recognizer-decoder/FP32/text-spotting-0005-recognizer-decoder.xml
    ... 100%, 38 KB, 296 KB/s, 0 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-recognizer-decoder/FP32/text-spotting-0005-recognizer-decoder.bin
    ... 100%, 2644 KB, 2323 KB/s, 1 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-recognizer-decoder/FP16/text-spotting-0005-recognizer-decoder.xml
    ... 100%, 45 KB, 332 KB/s, 0 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-recognizer-decoder/FP16/text-spotting-0005-recognizer-decoder.bin
    ... 100%, 1322 KB, 1630 KB/s, 0 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-recognizer-decoder/FP16-INT8/text-spotting-0005-recognizer-decoder.xml
    ... 100%, 66 KB, 263 KB/s, 0 seconds passed
    
    ========== Downloading models/intel/text-spotting-0005/text-spotting-0005-recognizer-decoder/FP16-INT8/text-spotting-0005-recognizer-decoder.bin
    ... 100%, 1059 KB, 2177 KB/s, 0 seconds passed
    



```python
from openvino.inference_engine import IECore

ie = IECore()
```


```python
devices = ie.available_devices
for device in devices:
    device_name = ie.get_metric(device_name=device, metric_name="FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")
```

    CPU: Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz


## <a class="anchor" id="2" style="color:black"><div style="padding:28px;color:black;margin:0;font-size:24px;background-color:gray;font-weight:bold;text-align:center;display:fill;border-radius:10px;overflow:hidden;line-height:36px;background:linear-gradient(100deg, gray, gray, gray, gray)"><span style="color:black">Iterate through outputs from designated classes '0123456789abcdefghijklmnopqrstuvwxyz#'</div></a>


```python
!mkdir text_spotting_demo/
```


```python
%%writefile text_spotting_demo/tracker.py

"""
 Copyright (c) 2019-2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np


class StaticIOUTracker(object):
    def __init__(self, iou_threshold=0.5, age_threshold=10):
        super().__init__()
        self.history = []
        self.history_areas = []
        self.history_classes = []
        self.ids = []
        self.age = []
        self.iou_threshold = iou_threshold
        self.age_threshold = age_threshold
        self.last_id = 0

    def affinity(self, masks, classes):
        areas = [np.count_nonzero(mask) for mask in masks]
        affinity_matrix = np.zeros((len(masks), len(self.history)), dtype=np.float32)
        for i, (history_mask, history_area, history_class) in \
                enumerate(zip(self.history, self.history_areas, self.history_classes)):
            for j, (mask, area, cls) in enumerate(zip(masks, areas, classes)):
                if cls != history_class:
                    continue
                intersection = np.count_nonzero(np.logical_and(history_mask, mask))
                union = history_area + area - intersection
                iou = intersection / union
                affinity_matrix[j, i] = iou
        return affinity_matrix, areas

    def __call__(self, masks, classes):
        # Get affinity with history.
        affinity_matrix, areas = self.affinity(masks, classes)

        # Make assignment of currents masks to existing tracks.
        assignment = []
        indices = np.arange(len(self.history))
        for i in range(len(masks)):
            j = 0
            affinity_score = -1.0
            if affinity_matrix.shape[1] > 0:
                j = np.argmax(affinity_matrix[i])
                affinity_score = affinity_matrix[i, j]
            if affinity_score > self.iou_threshold:
                assignment.append(indices[j])
                affinity_matrix = np.delete(affinity_matrix, j, 1)
                indices = np.delete(indices, j)
            else:
                assignment.append(None)

        # Increase age for existing tracks.
        for i in range(len(self.age)):
            self.age[i] += 1

        # Update existing tracks.
        for i, j in enumerate(assignment):
            if j is not None:
                self.history[j] = masks[i]
                self.history_areas[j] = areas[i]
                self.age[j] = 0
                assignment[i] = self.ids[j]

        # Prune out too old tracks.
        alive = tuple(i for i, age in enumerate(self.age) if age < self.age_threshold)
        self.history = [self.history[i] for i in alive]
        self.history_areas = [self.history_areas[i] for i in alive]
        self.history_classes = [self.history_classes[i] for i in alive]
        self.age = [self.age[i] for i in alive]
        self.ids = [self.ids[i] for i in alive]

        # Save new tracks.
        for i, j in enumerate(assignment):
            if j is None:
                self.history.append(masks[i])
                self.history_areas.append(areas[i])
                self.history_classes.append(classes[i])
                self.age.append(0)
                self.ids.append(self.last_id)
                assignment[i] = self.last_id
                self.last_id += 1

        return assignment
```

    Writing text_spotting_demo/tracker.py



```python
%%writefile text_spotting_demo/__init__.py


```

    Writing text_spotting_demo/__init__.py



```python
%%writefile text_spotting_demo.py


#!/usr/bin/env python3
"""
 Copyright (c) 2019-2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import sys
from pathlib import Path
from time import perf_counter
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np
from scipy.special import softmax
from openvino.runtime import Core, get_version

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

SOS_INDEX = 0
EOS_INDEX = 1
MAX_SEQ_LEN = 28


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m_m', '--mask_rcnn_model',
                      help='Required. Path to an .xml file with a trained Mask-RCNN model with '
                           'additional text features output.',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-m_te', '--text_enc_model',
                      help='Required. Path to an .xml file with a trained text recognition model '
                           '(encoder part).',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-m_td', '--text_dec_model',
                      help='Required. Path to an .xml file with a trained text recognition model '
                           '(decoder part).',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of the output file(s) to save.')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                           'If 0 is set, all frames are stored.')
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on, i.e : CPU, GPU. '
                           'The demo will look for a suitable plugin for device specified '
                           '(by default, it is CPU). Please refer to OpenVINO documentation '
                           'for the list of devices supported by the model.',
                      default='CPU', type=str, metavar='"<device>"')
    args.add_argument('--delay',
                      help='Optional. Interval in milliseconds of waiting for a key to be pressed.',
                      default=0, type=int, metavar='"<num>"')
    args.add_argument('-pt', '--prob_threshold',
                      help='Optional. Probability threshold for detections filtering.',
                      default=0.2, type=float, metavar='"<num>"')
    args.add_argument('-a', '--alphabet',
                      help='Optional. Alphabet that is used for decoding.',
                      default='  abcdefghijklmnopqrstuvwxyz0123456789')
    args.add_argument('--trd_input_prev_symbol',
                      help='Optional. Name of previous symbol input node to text recognition head decoder part.',
                      default='prev_symbol')
    args.add_argument('--trd_input_prev_hidden',
                      help='Optional. Name of previous hidden input node to text recognition head decoder part.',
                      default='prev_hidden')
    args.add_argument('--trd_input_encoder_outputs',
                      help='Optional. Name of encoder outputs input node to text recognition head decoder part.',
                      default='encoder_outputs')
    args.add_argument('--trd_output_symbols_distr',
                      help='Optional. Name of symbols distribution output node from text recognition head decoder part.',
                      default='output')
    args.add_argument('--trd_output_cur_hidden',
                      help='Optional. Name of current hidden output node from text recognition head decoder part.',
                      default='hidden')
    args.add_argument('-trt', '--tr_threshold',
                      help='Optional. Text recognition confidence threshold.',
                      default=0.001, type=float, metavar='"<num>"')
    args.add_argument('--keep_aspect_ratio',
                      help='Optional. Force image resize to keep aspect ratio.',
                      action='store_true')
    args.add_argument('--no_track',
                      help='Optional. Disable tracking.',
                      action='store_true')
    args.add_argument('--show_scores',
                      help='Optional. Show detection scores.',
                      action='store_true')
    args.add_argument('--show_boxes',
                      help='Optional. Show bounding boxes.',
                      action='store_true')
    args.add_argument('-r', '--raw_output_message',
                      help='Optional. Output inference results raw values.',
                      action='store_true')
    args.add_argument("--no_show",
                      help="Optional. Don't show output",
                      action='store_true')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    return parser


def expand_box(box, scale):
    w_half = (box[2] - box[0]) * .5
    h_half = (box[3] - box[1]) * .5
    x_c = (box[2] + box[0]) * .5
    y_c = (box[3] + box[1]) * .5
    w_half *= scale
    h_half *= scale
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp


def segm_postprocess(box, raw_cls_mask, im_h, im_w):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)
    # Put an object mask in an image mask.
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                            (x0 - extended_box[0]):(x1 - extended_box[0])]
    return im_mask


def main():
    args = build_argparser().parse_args()

    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    # Read IR
    log.info('Reading Mask-RCNN model {}'.format(args.mask_rcnn_model))
    mask_rcnn_model = core.read_model(args.mask_rcnn_model)

    input_tensor_name = 'image'
    try:
        n, c, h, w = mask_rcnn_model.input(input_tensor_name).shape
        if n != 1:
            raise RuntimeError('Only batch 1 is supported by the demo application')
    except RuntimeError:
        raise RuntimeError('Demo supports only topologies with the following input tensor name: {}'.format(input_tensor_name))

    required_output_names = {'boxes', 'labels', 'masks', 'text_features'}
    for output_tensor_name in required_output_names:
        try:
            mask_rcnn_model.output(output_tensor_name)
        except RuntimeError:
            raise RuntimeError('Demo supports only topologies with the following output tensor names: {}'.format(
                ', '.join(required_output_names)))

    log.info('Reading Text Recognition Encoder model {}'.format(args.text_enc_model))
    text_enc_model = core.read_model(args.text_enc_model)

    log.info('Reading Text Recognition Decoder model {}'.format(args.text_dec_model))
    text_dec_model = core.read_model(args.text_dec_model)

    mask_rcnn_compiled_model = core.compile_model(mask_rcnn_model, device_name=args.device)
    mask_rcnn_infer_request = mask_rcnn_compiled_model.create_infer_request()
    log.info('The Mask-RCNN model {} is loaded to {}'.format(args.mask_rcnn_model, args.device))

    text_enc_compiled_model = core.compile_model(text_enc_model, args.device)
    text_enc_output_tensor = text_enc_compiled_model.outputs[0]
    text_enc_infer_request = text_enc_compiled_model.create_infer_request()
    log.info('The Text Recognition Encoder model {} is loaded to {}'.format(args.text_enc_model, args.device))

    text_dec_compiled_model = core.compile_model(text_dec_model, args.device)
    text_dec_infer_request = text_dec_compiled_model.create_infer_request()
    log.info('The Text Recognition Decoder model {} is loaded to {}'.format(args.text_dec_model, args.device))

    hidden_shape = text_dec_model.input(args.trd_input_prev_hidden).shape
    text_dec_output_names = {args.trd_output_symbols_distr, args.trd_output_cur_hidden}
    
    log.info('Text decoder output names are {}'.format(text_dec_output_names))

    frames_processed = 0

    start_time = perf_counter()
    frame = image
    
    print(frame.shape)

    if not args.keep_aspect_ratio:
        # Resize the image to a target size.
        scale_x = w / frame.shape[1]
        scale_y = h / frame.shape[0]
        input_image = cv2.resize(frame, (w, h))
    else:
        # Resize the image to keep the same aspect ratio and to fit it to a window of a target size.
        scale_x = scale_y = min(h / frame.shape[0], w / frame.shape[1])
        input_image = cv2.resize(frame, None, fx=scale_x, fy=scale_y)

    print(input_image.shape)
    
    input_image_size = input_image.shape[:2]
    input_image = np.pad(input_image, ((0, h - input_image_size[0]),
                                       (0, w - input_image_size[1]),
                                       (0, 0)),
                         mode='constant', constant_values=0)
    # Change data layout from HWC to CHW.
    input_image = input_image.transpose((2, 0, 1))
    input_image = input_image.reshape((n, c, h, w)).astype(np.float32)
    
    print(input_image.shape)

    # Run the MaskRCNN model.
    mask_rcnn_infer_request.infer({input_tensor_name: input_image})
    outputs = {name: mask_rcnn_infer_request.get_tensor(name).data[:] for name in required_output_names}
    
#     log.info('Outputs are {}'.format(outputs))

    # Parse detection results of the current request
    boxes = outputs['boxes'][:, :4]
    scores = outputs['boxes'][:, 4]
    classes = outputs['labels'].astype(np.uint32)
    raw_masks = outputs['masks']
    text_features = outputs['text_features']
    
#     log.info('Text Features are {}'.format(text_features))

    # Filter out detections with low confidence.
    detections_filter = scores > args.prob_threshold
    scores = scores[detections_filter]
    classes = classes[detections_filter]
    boxes = boxes[detections_filter]
    raw_masks = raw_masks[detections_filter]
    text_features = text_features[detections_filter]

    boxes[:, 0::2] /= scale_x
    boxes[:, 1::2] /= scale_y
    masks = []
    for box, cls, raw_mask in zip(boxes, classes, raw_masks):
        mask = segm_postprocess(box, raw_mask, frame.shape[0], frame.shape[1])
        masks.append(mask)

    texts = []
    for feature in text_features:
        input_data = {'input': np.expand_dims(feature, axis=0)}
        feature = text_enc_infer_request.infer(input_data)[text_enc_output_tensor]
        feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
        feature = np.transpose(feature, (0, 2, 1))

        hidden = np.zeros(hidden_shape)
        prev_symbol_index = np.ones((1,)) * SOS_INDEX

        text = ''
        text_confidence = 1.0
        for i in range(MAX_SEQ_LEN):
            text_dec_infer_request.infer({
                args.trd_input_prev_symbol: np.reshape(prev_symbol_index, (1,)),
                args.trd_input_prev_hidden: hidden,
                args.trd_input_encoder_outputs: feature})
            decoder_output = {name: text_dec_infer_request.get_tensor(name).data[:] for name in text_dec_output_names}
            symbols_distr = decoder_output[args.trd_output_symbols_distr]
            symbols_distr_softmaxed = softmax(symbols_distr, axis=1)[0]
            prev_symbol_index = int(np.argmax(symbols_distr, axis=1))
            text_confidence *= symbols_distr_softmaxed[prev_symbol_index]
            if prev_symbol_index == EOS_INDEX:
                break
            text += args.alphabet[prev_symbol_index]
            hidden = decoder_output[args.trd_output_cur_hidden]

        texts.append(text if text_confidence >= args.tr_threshold else '')

    log.info("Text confidence is {}".format(text_confidence))
    
    log.info('Text Features are {}'.format(texts))
    
    if len(boxes) and args.raw_output_message:
        log.debug('  -------------------------- Frame # {} --------------------------  '.format(frames_processed))
        log.debug('  Class ID | Confidence |     XMIN |     YMIN |     XMAX |     YMAX ')
        for box, cls, score, mask in zip(boxes, classes, scores, masks):
            log.debug('{:>10} | {:>10f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} '.format(cls, score, *box))

    start_time = perf_counter()

if __name__ == '__main__':
    sys.exit(main() or 0)
```

    Overwriting text_spotting_demo.py



```python
%%writefile text_recognition_demo.sh

python3 /home/u186775/image-recognition-with-privacy/text_spotting_demo.py \
-m_m "/home/u186775/image-recognition-with-privacy/models/intel/text-spotting-0005/text-spotting-0005-detector/FP16/text-spotting-0005-detector.xml" \
-m_te "/home/u186775/image-recognition-with-privacy/models/intel/text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-encoder.xml" \
-m_td "/home/u186775/image-recognition-with-privacy/models/intel/text-spotting-0005/text-spotting-0005-recognizer-decoder/FP16/text-spotting-0005-recognizer-decoder.xml" \
-i "/home/u186775/image-recognition-with-privacy/inference_images/text_inference.png"
```

    Overwriting text_recognition_demo.sh



```python
display.Image("/home/u186775/image-recognition-with-privacy/inference_images/text_inference.png")
```




    
![png](output_202_0.png)
    




```python
!qsub text_recognition_demo.sh -o /home/u186775/image-recognition-with-privacy/logs/text_spotting_log.out -e /home/u186775/image-recognition-with-privacy/logs/text_spotting_log.err
```

    2231308.v-qsvr-1.aidevcloud



```python
!qstat
```

    Job ID                    Name             User            Time Use S Queue
    ------------------------- ---------------- --------------- -------- - -----
    2231261.v-qsvr-1           ...ub-singleuser u186775         00:03:57 R jupyterhub     



```python
!cat logs/text_spotting_log.out
```

    
    ########################################################################
    #      Date:           Sat 04 Mar 2023 09:44:18 PM PST
    #    Job ID:           2231308.v-qsvr-1.aidevcloud
    #      User:           u186775
    # Resources:           cput=75:00:00,neednodes=1:batch:ppn=2,nodes=1:batch:ppn=2,walltime=06:00:00
    ########################################################################
    
    [ INFO ] OpenVINO Runtime
    [ INFO ] 	build: 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] Reading Mask-RCNN model /home/u186775/image-recognition-with-privacy/models/intel/text-spotting-0005/text-spotting-0005-detector/FP16/text-spotting-0005-detector.xml
    [ INFO ] Reading Text Recognition Encoder model /home/u186775/image-recognition-with-privacy/models/intel/text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-encoder.xml
    [ INFO ] Reading Text Recognition Decoder model /home/u186775/image-recognition-with-privacy/models/intel/text-spotting-0005/text-spotting-0005-recognizer-decoder/FP16/text-spotting-0005-recognizer-decoder.xml
    [ INFO ] The Mask-RCNN model /home/u186775/image-recognition-with-privacy/models/intel/text-spotting-0005/text-spotting-0005-detector/FP16/text-spotting-0005-detector.xml is loaded to CPU
    [ INFO ] The Text Recognition Encoder model /home/u186775/image-recognition-with-privacy/models/intel/text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-encoder.xml is loaded to CPU
    [ INFO ] The Text Recognition Decoder model /home/u186775/image-recognition-with-privacy/models/intel/text-spotting-0005/text-spotting-0005-recognizer-decoder/FP16/text-spotting-0005-recognizer-decoder.xml is loaded to CPU
    [ INFO ] Text decoder output names are {'hidden', 'output'}
    (573, 820, 3)
    (768, 1280, 3)
    (1, 3, 768, 1280)
    [ INFO ] Text confidence is 0.13285909327739348
    [ INFO ] Text Features are ['3', '2', '7', '4', '8', '9', 'fors', 'wayz', '6', 'tuv', 'ghi', 'def', 'lo1', '5', 'aic', 'mnd', 'eci']
    
    ########################################################################
    # End of output for job 2231308.v-qsvr-1.aidevcloud
    # Date: Sat 04 Mar 2023 09:44:42 PM PST
    ########################################################################
    



```python
total_identification.replace(" ", "").split(), total_original.replace(" ", "").split()
```




    (['327489forswayz6tuvghideflo15aicmndeci'],
     ['327489pqrswxyz6tuvghidefjkl5abcmnoghi'])




```python
from sklearn.metrics import accuracy_score

identification = ['3', '2', '7', '4', '8', '9', 'fors', 'wayz', '6', 'tuv', 'ghi', 'def', 'lo1', '5', 'aic', 'mnd', 'eci']
original = ['3', '2', '7', '4', '8', '9', 'pqrs', 'wxyz', '6', 'tuv', 'ghi', 'def', 'jkl', '5', 'abc', 'mno', 'ghi']

total_identification = " ".join(identification)
total_original = " ".join(original)

print("""Identification: {} characters, Original: {} characters, Accuracy: {}""".format(len(total_identification.replace(" ", "")), 
                                                                                        len(total_original.replace(" ", "")), 
                                                                                       accuracy_score(list(total_identification.replace(" ", "")), 
                                                                                                     list(total_original.replace(" ", "")))))
```

    Identification: 37 characters, Original: 37 characters, Accuracy: 0.7297297297297297


**Accuracy: 0.7297297297297297 for 37 characters**
====================================================

# **Summary and Findings**

## Text Recognition:

- First model I used was NNCF (Neural Network Compression Framework) with a Pytorch based Model called, CRNN.PYTORCH a CNN-LSTM based model, which was found to show a hge error
- Next I used PaddleOCR Model from the PaddlePaddle Project, which gave good results on the native framework, but did not produce effective results for the strip data based images dataset, by importing the paddle model into OpenVINO
- Then I used MMOCR Model, from the OpenMMLab Library, which gave good results on the native framework, but had a CUDA dependency that was having a certain degree of complexity
- In Another trial, I used EasyOCR Model, which is a PyTorch based model, which gave good results on the native framework, but gave me confidence threshold based results with the PyTorch nn.Module Class with in-consistent results, for the same sample image
- So finally I proceeded with an OpenVINO Model that performed good for a different sample image

## Number Plate Recognition:

- For this, I used a RoboFlow based model with Remote APIs and API Key

## Object Detection:

- For this, I used a Pre-Trained RoboFlow model, which was trained by me, using 900+ images from Construction Site

## **Code on a Separate Github Repository from EDA Code**

<a class="anchor" id="2" style="color:white;"><div style="padding:28px;color:white;margin:0;font-size:52px;font-weight:bold;text-align:left;display:fill;border-radius:10px;background-color:#00022f;overflow:hidden;line-height:64px;background:linear-gradient(100deg, navy, #251cab, #3eb4f4, #251cab)">**Reference:**<br/><br/>
https://github.com/aswinvk28/image-recognition-with-privacy </div></a>
