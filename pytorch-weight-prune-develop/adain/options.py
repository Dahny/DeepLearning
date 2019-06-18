import argparse


class Options:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # Basic options
        self.parser.add_argument('--content', type=str,
                                 help='File path to the content image')
        self.parser.add_argument('--content_dir', type=str,
                                 help='Directory path to a batch of content images')
        self.parser.add_argument('--style', type=str,
                                 help='File path to the style image, or multiple style \
                               images separated by commas if you want to do style \
                               interpolation or spatial control')
        self.parser.add_argument('--style_dir', type=str,
                                 help='Directory path to a batch of style images')
        self.parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
        self.parser.add_argument('--decoder', type=str, default='models/decoder.pth')

        # Additional options
        self.parser.add_argument('--content_size', type=int, default=512,
                                 help='New (minimum) size for the content image, \
                               keeping the original size if set to 0')
        self.parser.add_argument('--style_size', type=int, default=512,
                                 help='New (minimum) size for the style image, \
                               keeping the original size if set to 0')
        self.parser.add_argument('--crop', action='store_true',
                                 help='do center crop to create squared image')
        self.parser.add_argument('--save_ext', default='.jpg',
                                 help='The extension name of the output image')
        self.parser.add_argument('--output', type=str, default='outputNe',
                                 help='Directory to save the output image(s)')

        # Advanced options
        self.parser.add_argument('--preserve_color', action='store_true',
                                 help='If specified, preserve color of the content image')
        self.parser.add_argument('--alpha', type=float, default=1.0,
                                 help='The weight that controls the degree of \
                                        stylization. Should be between 0 and 1')
        self.parser.add_argument(
            '--style_interpolation_weights', type=str, default='',
            help='The weight for blending the style of multiple style images')

	# training options
        self.parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
        self.parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--lr_decay', type=float, default=5e-5)
        self.parser.add_argument('--max_iter', type=int, default=160000)
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--style_weight', type=float, default=10.0)
        self.parser.add_argument('--content_weight', type=float, default=1.0)
        self.parser.add_argument('--n_threads', type=int, default=16)
        self.parser.add_argument('--save_model_interval', type=int, default=10000)

    def parse(self):
        return self.parser.parse_args()

    def test_arg(self):
        args = self.parser.parse_args(['--content_dir', '../pytorch-AdaIN-master/input/content',
                                  '--style_dir', '../pytorch-AdaIN-master/input/style'
                                  ])
        return args

    def train_arg(self):
        args = self.parser.parse_args(['--content_dir', './data/MSCOCO/val2017',
                                       '--style_dir', '../pytorch-AdaIN-master/input/style',
                                       '--save_dir', './models'
                                       ])
        return args
