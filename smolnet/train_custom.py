import sys
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from unet_model import UNet
from unet_model_smaller import UNet as SmolNet
from styletransferdataset import StyleTransferDataset, ToTensor, Crop
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import time

def show_img(img):
    plt.imshow(np.transpose(img.detach().cpu().numpy()[0], (1, 2, 0)))
    plt.show()

def train_net(
        net,
        dataset,
        device,
        epochs=5,
        batch_size=1,
        lr=0.0000001):

    criterion = nn.MSELoss(reduction='sum')
    # optimizer = optimizer = optim.SGD(net.parameters(),
    #                       lr=lr,
    #                       momentum=0.9,
    #                       weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    for epoch in range(epochs):

        running_loss = 0.0
        print('epoch', epoch)

        for i, data in enumerate(dataset):

            #print('data', data)

            input = data['input']
            target = data['target']

            try:
                input = input.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                pre = time.time()
                output = net(input)
                delta = time.time() - pre
                #cprint('Forward pass: ', delta)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            except RuntimeError:
                print(i)
                print(input.shape)
                del(input) # Delete problematic input variable so memory can be freed
                print('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
                print('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
                print('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))
                print("Skipping input because OUT OF MEMORY")
                torch.cuda.empty_cache()
                time.sleep(1)
                continue



            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 2000 mini-batches

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
                show_img(output)

            # show result
            #show_img(input)
            #show_img(target)




if __name__ == '__main__':
    net = UNet(n_input_channels=3, n_output_channels=3)
    #net = SmolNet(n_input_channels=3, n_output_channels=3)
    # dataset = Subset(
    #     StyleTransferDataset('dataset/training', transform=ToTensor()),
    #     range(1)
    # )

    dataset = StyleTransferDataset('dataset/training',
                                   transform=transforms.Compose([
                                       Crop(),
                                       ToTensor()
                                   ]))
    dataloader = DataLoader(dataset, batch_size=1,
                                 shuffle=True, num_workers=4)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net.to(device)

    try:
        train_net(
            net=net,
            dataset=dataloader,
            device=device,
            epochs=1000
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)