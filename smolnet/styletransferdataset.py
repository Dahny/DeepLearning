from pathlib import Path
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
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
        target_path = f"{re.search('(.*).jpg', input_path).group(1)}{self.outputpostfix}".replace('/input/', '/output/')

        input_image = io.imread(input_path)
        target_image = io.imread(target_path)

        sample = {'input': input_image, 'target': target_image}

        if self.transform:
            sample = self.transform(sample)

        return sample