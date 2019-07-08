import torch.nn as nn
from pruning.layers import MaskedConv2d
from adain.function import *

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    MaskedConv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    MaskedConv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class AdainNet(nn.Module):
    def __init__(self, encoder, decoder):
        super(AdainNet, self).__init__()

        self.enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*self.enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*self.enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*self.enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*self.enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.dec_layers = list(decoder.children())
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

        # extract relu4_1 from input image

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adaptive_instance_normalization(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s

    def set_enc_masks(self, masks):
        # Apply mask for every MaskedConv2d in encoder
        # the last layer used in encoder is layer 31 so higher layers don't take into account
        self.enc_layers[0].set_mask(masks[0])
        self.enc_layers[2].set_mask(masks[1])
        self.enc_layers[5].set_mask(masks[2])
        self.enc_layers[9].set_mask(masks[3])
        self.enc_layers[12].set_mask(masks[4])
        self.enc_layers[16].set_mask(masks[5])
        self.enc_layers[19].set_mask(masks[6])
        self.enc_layers[22].set_mask(masks[7])
        self.enc_layers[25].set_mask(masks[8])
        self.enc_layers[29].set_mask(masks[9])

    def set_dec_masks(self, masks):
        # Apply mask for every MaskedConv2d in decoder
        self.dec_layers[1].set_mask(masks[0])
        self.dec_layers[5].set_mask(masks[1])
        self.dec_layers[8].set_mask(masks[2])
        self.dec_layers[11].set_mask(masks[3])
        self.dec_layers[14].set_mask(masks[4])
        self.dec_layers[18].set_mask(masks[5])
        self.dec_layers[21].set_mask(masks[6])
        self.dec_layers[25].set_mask(masks[7])
        self.dec_layers[28].set_mask(masks[8])




