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

times = []

with torch.no_grad():
    net = UNet(n_input_channels=3, n_output_channels=3)
    net.load_state_dict(torch.load('G_unet_sketch.pth'))
    net.eval()
    net.to(device)

    print(f'Number of params: {sum(p.numel() for p in net.parameters())}')
    print(f'Number of trainable params: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')

    for impath in list(Path('../dataset/matisse/input').glob('*.jpg'))[0:100]:
        name = impath.name
        impath = str(impath)
        inputimg = io.imread(impath).astype(np.float32) / 255
        inputimg = inputimg.transpose((2, 0, 1))
        inputimg = torch.from_numpy(inputimg).unsqueeze(0)
        inputimg = inputimg.to(device)
        #print(inputimg.shape)

        before = time.time()
        outputimg = net(inputimg)
        delta = time.time() - before

        times.append(delta)

        #show_img(outputimg, name)

print(np.mean(np.array(times)), np.std(np.array(times)))