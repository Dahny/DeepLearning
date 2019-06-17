import numpy as np
import torch
from os.path import basename
from os.path import splitext
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler
from adain.function import *
import torch.utils.data as data
import time
from adain.options import Options
from adain.test_function import *
from torchvision.utils import save_image

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


def test_adain(vgg, decoder):

    begin_time = time.time()

    # read parameters
    args = Options().test_arg()
    do_interpolation, device, content_paths, style_paths, interpolation_weights = configuration(args)

    # set models in evaluation mode
    decoder.eval()
    vgg.eval()
    vgg.to(device)
    decoder.to(device)

    content_tf = test_transform(args.content_size, args.crop)
    style_tf = test_transform(args.style_size, args.crop)

    # For some reason the first style transfer takes up a whole extra second
    # which ruines the data so we do the first run twice :)
    first_time = True

    # 5 styles and 300 content images
    benchmarks = np.ndarray([5, 300])

    for (idxContent, content_path) in enumerate(content_paths):
        if do_interpolation:  # one content image, N style image
            style = torch.stack([style_tf(Image.open(p)) for p in style_paths])
            content = content_tf(Image.open(content_path)) \
                .unsqueeze(0).expand_as(style)

            # content = content_tf(Image.open(content)) \
            #    .unsqueeze(0).expand_as(style)
            style = style.to(device)
            content = content.to(device)

            begin_time2 = time.time()

            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha, interpolation_weights)

            end_time2 = time.time()

            # Set timing benchmark
            benchmarks[idxStyle, idxContent] = end_time2 - begin_time2

            output = output.cpu()
            output_name = '{:s}/{:s}_interpolation{:s}'.format(
                args.output, splitext(basename(content_path))[0], args.save_ext)
            save_image(output, output_name)

        else:  # process one content and one style
            for (idxStyle, style_path) in enumerate(style_paths):
                content = content_tf(Image.open(content_path))
                style = style_tf(Image.open(style_path))
                if args.preserve_color:
                    style = coral(style, content)
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)

                # Begin time
                begin_time2 = time.time()

                with torch.no_grad():
                    output = style_transfer(vgg, decoder, content, style,
                                            args.alpha)
                # End time
                end_time2 = time.time()

                # Do the first run twice to fix first run extra second bug
                # SORRY FOR THE VERY UGLY WAY TO DO THIS :(
                if first_time & idxContent == 0 & idxStyle == 0:
                    # Begin time
                    begin_time2 = time.time()

                    with torch.no_grad():
                        output = style_transfer(vgg, decoder, content, style,
                                                args.alpha)
                    # End time
                    end_time2 = time.time()

                    first_time = False

                # Set timing benchmark
                benchmarks[idxStyle, idxContent] = end_time2 - begin_time2

                output = output.cpu()

                output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                    args.output, splitext(basename(content_path))[0],
                    splitext(basename(style_path))[0], args.save_ext
                )
                save_image(output, output_name)

    end_time = time.time();
    print("The total amount of time for test is", end_time - begin_time, "with the setup:", device.type)

    if do_interpolation:  # one content image, N style image
        write_benchmark(benchmarks)


def write_benchmark(benchmarks):
    means = []
    stds = []

    f = open("benchmark.txt", "a")
    # for each Style
    for idx1, style in enumerate(benchmarks):
        print("Style", idx1)
        f.write("Style " + str(idx1) + "\n")

        # For each content image of this style
        for idx2, item in enumerate(style):
            print("The time for item", idx2, "took:", item, "seconds")
            f.writelines("frame " + str(idx2) + " took " + str(item) + " seconds\n")

        # Compute avg and std for the style
        styleSTD = np.std(style)
        print("The Standard Deviation for style", idx1, "is", styleSTD)
        stds.append(styleSTD)

        styleAVG = np.mean(style)
        print("The Average for style", idx1, "is", styleAVG)
        means.append(styleAVG)

    # Write the style avgs and stds
    for i in range(5):
        f.write("Style " + str(i) + " avg: " + str(means[i]) + "\n")
        f.write("Style " + str(i) + " std: " + str(stds[i]) + "\n")

    totalAVG = np.mean(benchmarks)
    totalSTD = np.std(benchmarks)
    print("The total average time took:", totalAVG, "seconds\n")
    f.write("total_avg:" + str(totalAVG) + "\n")

    print("The total standard deviation:", totalSTD)
    f.write("std:" + str(totalSTD) + "\n")

    f.close()

    return totalAVG, totalSTD


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
    
