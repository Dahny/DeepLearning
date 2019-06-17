"""
Pruning a MLP by weights with one shot
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import adainNet

from pruning.methods import weight_prune
from pruning.utils import to_var, train, test, prune_rate
from models import MLP


# Hyper Parameters
param = {
    'pruning_perc': 90.,
    'batch_size': 128,
    'test_batch_size': 100,
    'num_epochs': 5,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
}


# Data loaders
# train_dataset = datasets.MNIST(root='../data/',train=True, download=True,
#     transform=transforms.ToTensor())
# loader_train = torch.utils.data.DataLoader(train_dataset,
#     batch_size=param['batch_size'], shuffle=True)
#
# test_dataset = datasets.MNIST(root='../data/', train=False, download=True,
#     transform=transforms.ToTensor())
# loader_test = torch.utils.data.DataLoader(test_dataset,
#     batch_size=param['test_batch_size'], shuffle=True)


# Load the pretrained model

vgg = adainNet.vgg
decoder = adainNet.decoder
vgg.load_state_dict("../pytorch-AdaIN-master/models/vgg_normalised.pth")
decoder.load_state_dict("../pytorch-AdaIN-master/models/decoder.pth")
net = adainNet.AdainNet(vgg, decoder)

if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()
print("--- Pretrained network loaded ---")
# test(net, loader_test)

# prune the weights
masks = weight_prune(net, param['pruning_perc'])
net.set_masks(masks)
net = nn.DataParallel(net).cuda()
print("--- {}% parameters pruned ---".format(param['pruning_perc']))
# test(net, loader_test)


# Retraining
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
#                                 weight_decay=param['weight_decay'])
#
# train(net, criterion, optimizer, param, loader_train)


# Check accuracy and nonzeros weights in each layer
print("--- After retraining ---")
# test(net, loader_test)
prune_rate(net)


# Save and load the entire model
torch.save(vgg.state_dict(), 'model/vgg_pruned.pkl')
torch.save(decoder.state_dict(), 'model/decoder_pruned.pkl')