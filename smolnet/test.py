import sys
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from unet_model import UNet
from unet_model_smaller import UNet as SmolNet
from styletransferdataset import StyleTransferDataset, ToTensor
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import time
from skimage import io
from pathlib import Path

def show_img(img, name):
    plt.axis('off')
    plt.imshow(np.transpose(img.detach().cpu().numpy()[0], (1, 2, 0)))
    plt.savefig(name, bbox_inches='tight')
    plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    net = UNet(n_input_channels=3, n_output_channels=3)
    net.load_state_dict(torch.load('G_unet_sketch.pth'))
    net.to(device)
    net.eval()

    for impath in Path('../dataset/nick/input').glob('*.jpg'):
        name = impath.name
        impath = str(impath)
        inputimg = io.imread(impath).astype(np.float32) / 255
        inputimg = inputimg.transpose((2, 0, 1))
        inputimg = torch.from_numpy(inputimg).unsqueeze(0)
        inputimg = inputimg.to(device)
        print(inputimg.shape)

        outputimg = net(inputimg)

        show_img(outputimg, name)
