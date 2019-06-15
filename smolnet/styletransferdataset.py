from pathlib import Path
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import numpy as np
import matplotlib.pyplot as plt
import re

class StyleTransferDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = Path(dir)
        self.inputdir = self.dir.joinpath('input')
        self.targetdir = self.dir.joinpath('output')

        self.inputimages = list(self.inputdir.glob('*.jpg'))
        self.targetimages = list(self.targetdir.glob('*.jpg'))

        self.outputpostfix = re.search('[0-9]*(.*)', str(self.targetimages[0].name)).group(1)

        self.transform = transform


    def __len__(self):
        return len(self.inputimages)

    def __getitem__(self, idx):
        input_path = str(self.inputimages[idx])

        target_path_parts = list(Path(f"{re.search('(.*).jpg', input_path).group(1)}{self.outputpostfix}").parts)
        target_path_parts[target_path_parts.index('input')] = 'output'
        target_path = Path(*target_path_parts)

        input_image = io.imread(input_path)
        target_image = io.imread(target_path)

        sample = {'input': input_image, 'target': target_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input, target = sample['input'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input = input.transpose((2, 0, 1))
        target = target.transpose((2, 0, 1))

        # Convert to float
        input = input.astype(np.float32) / 255
        target = target.astype(np.float32) / 255

        #print('input', input.shape)
        #print('target', target.shape)

        return {'input': torch.from_numpy(input),
                'target': torch.from_numpy(target)}

if __name__ == "__main__":
    def imshow(img):
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.show()


    dataset = StyleTransferDataset('dataset/training', transform=ToTensor())

    dataloader = DataLoader(dataset, batch_size=1,
                                          shuffle=True, num_workers=4)
    tmp = next(iter(dataloader))
    input = tmp['input']
    target = tmp['target']

    imshow(utils.make_grid(input))
    imshow(utils.make_grid(target))