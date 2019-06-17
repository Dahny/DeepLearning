import os
from torchvision import transforms
import torch
from adain.function import adaptive_instance_normalization


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def get_style_path(args):
    do_interpolation, interpolation_weights = None, None
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

    return style_paths, do_interpolation, interpolation_weights


def get_content_path(args):
    if args.content:
        content_paths = [args.content]
    else:
        content_paths = [os.path.join(args.content_dir, f) for f in
                         os.listdir(args.content_dir)]
    return content_paths


def configuration(args):
    do_interpolation = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Either --content or --contentDir should be given.
    assert (args.content or args.content_dir)
    # Either --style or --styleDir should be given.
    assert (args.style or args.style_dir)

    content_paths = get_content_path(args)
    style_paths, do_interpolation, interpolation_weights = get_style_path(args)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    return (do_interpolation, device, content_paths, style_paths, interpolation_weights)


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
