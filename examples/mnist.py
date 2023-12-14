
#import torch
#from torch.utils.data.dataset import Dataset
#
#model MNist:
#.  input_flatten = 
#   hidden = input * ?ToNodes(10)
#   prediction = hidden * ?ToNodes(1)
#.  label ?= prediction

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

MNIST(root='MNIST_data/',train=True,transform=ToTensor(),download = True)
MNIST(root='MNIST_data/',train=False,transform=ToTensor(),download = True)