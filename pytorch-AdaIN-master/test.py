import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import time

import net
from function import adaptive_instance_normalization
from function import coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

begin_time = time.time();

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='outputNe',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)

if args.content:
    content_paths = [args.content]
else:
    content_paths = [os.path.join(args.content_dir, f) for f in
                     os.listdir(args.content_dir)]

if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [args.style]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_paths = [os.path.join(args.style_dir, f) for f in
                   os.listdir(args.style_dir)]

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

begin_time2 = 0
end_time2 = 0

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

        #content = content_tf(Image.open(content)) \
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

            #Begin time
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

means = []
stds = []

total = 0
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
    print("The Standard Deviation for style", idx1, "is",styleSTD)
    stds.append(styleSTD)

    styleAVG = np.mean(style)
    print("The Average for style", idx1, "is",styleAVG)
    means.append(styleAVG)


# Write the style avgs and stds
for i in range(5):
    f.write("Style " + str(i) + " avg: " + str(means[i]) + "\n")
    f.write("Style " + str(i) + " std: " + str(stds[i]) + "\n")

totalAVG = np.mean(benchmarks)
totalSTD = np.std(benchmarks)
print("The total average time took:", totalAVG, "seconds TJOM CJATSHJOEK\n")
f.write("total_avg:" + str(totalAVG) + "\n")

print("The total standard deviation:", totalSTD)
f.write("std:" + str(totalSTD) + "\n")

f.close()