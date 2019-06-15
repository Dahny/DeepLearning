import numpy as np
import torch
import os
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler
from torchvision import transforms
from adain.function import *
import torch.utils.data as data

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)
def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)
    
def train(model, loss_fn, optimizer, param, loader_train, loader_val=None):

    model.train()
    for epoch in range(param['num_epochs']):
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

        for t, (x, y) in enumerate(loader_train):
            x_var, y_var = to_var(x), to_var(y.long())

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            if (t + 1) % 100 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         

def train_adain(model, loss_fn, optimizer, param, loader_train,args, loader_val=None):
    device = torch.device('cuda')
    model.train()
    model.to(device)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    for epoch in range(param['num_epochs']):
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))
        adjust_learning_rate(optimizer,epoch,args.lr, args.lr_decay )
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)
        loss_c, loss_s = model(content_images, style_images)

        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, loader):

    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)
    for x, y in loader:
        x_var = to_var(x, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    print('Test accuracy: {:.2f}% ({}/{})'.format(
        100.*acc,
        num_correct,
        num_samples,
        ))
    
    return acc
    

def prune_rate(model, verbose=True):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

            if verbose:
                print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                        layer_id,
                        'Conv' if len(parameter.data.size()) == 4 \
                            else 'Linear',
                        100.*zero_param_this_layer/param_this_layer,
                        ))
    pruning_perc = 100.*nb_zero_param/total_nb_param
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc


def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
         if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix
    
