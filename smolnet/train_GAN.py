import sys
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from unet_model import UNet
from unet_model_smaller import UNet as SmolNet
from patch_discriminator import NLayerDiscriminator
from styletransferdataset import StyleTransferDataset, ToTensor, Crop
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn.functional as F

def show_img(og, target, g):
    plt.axis('off')
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 15))
    axes[0].imshow(np.transpose(og.detach().cpu().numpy()[0], (1, 2, 0)))
    axes[1].imshow(np.transpose(target.detach().cpu().numpy()[0], (1, 2, 0)))
    axes[2].imshow(np.transpose(g.detach().cpu().numpy()[0], (1, 2, 0)))
    plt.show()

def train_net(
        G,
        D,
        G_optimizer,
        D_optimizer,
        dataset,
        device,
        criterion,
        epochs=5):

    for epoch in range(epochs):

        # Keep track of running errors
        running_dre, running_dfe, running_ge = [0.0] * 3

        print('epoch', epoch)

        for i, data in enumerate(dataset):

            input = data['input']
            target = data['target']

            try:
                # Unpack data from dataset
                input = input.to(device)
                target = target.to(device)

                ### Step discriminator
                D_optimizer.zero_grad()

                # Train on real data
                D_real_data = target
                D_real_decision = D(D_real_data)
                D_real_error = criterion(D_real_decision, torch.ones(D_real_decision.shape).to(device))
                #D_real_error.backward()

                # Train on fake data
                D_fake_data = G(input).detach() # Detach to not backprop into G
                D_fake_decision = D(D_fake_data)
                D_fake_error = criterion(D_fake_decision, torch.zeros(D_fake_decision.shape).to(device))

                D_total_error = 0.5 * (D_real_error + D_fake_error) # Discriminator is OP, nerf pls
                D_total_error.backward()

                D_optimizer.step()

                ## Step Generator
                G_optimizer.zero_grad()

                # Generate fake output
                G_input = input
                G_generated = G(G_input)

                # Can the discriminator see the difference?
                DG_decision = D(G_generated)
                G_error_GAN = criterion(DG_decision, torch.ones(DG_decision.size()).to(device))

                # Also include L1 loss for low frequency content and preventing color fuckery
                #G_error_L1 = 0.1 * F.l1_loss(G_generated, target)#, reduction='sum')
                G_error_L1 = 0.0001 * F.mse_loss(G_generated, target, reduction='sum')

                G_error = G_error_GAN + G_error_L1

                # Train the generator to fool the discriminator
                G_error.backward()
                G_optimizer.step()

            except RuntimeError as e:
                print(i)
                print(input.shape)
                del(input) # Delete problematic input variable so memory can be freed
                print('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
                print('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
                print('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))
                print(f'Skipping input because {e}')
                torch.cuda.empty_cache()
                time.sleep(1)
                continue



            # print statistics
            running_dre += D_real_error.item()
            running_dfe += D_fake_error.item()
            running_ge += G_error.item()

            if i % 20 == 19:  # print every 20 mini-batches

                print('[%d, %5d] dre: %.3f, dfe: %.3f, ge: %.3f' %
                      (epoch + 1, i + 1, running_dre / 20, running_dfe / 20, running_ge / 20))
                show_img(input, target, G_generated)
                running_dre = 0
                running_dfe = 0
                running_ge = 0

            # show result
            #show_img(input)
            #show_img(target)




if __name__ == '__main__':
    G = UNet(n_input_channels=3, n_output_channels=3)
    #G.load_state_dict(torch.load('G_smolnet.pth'))

    #G = SmolNet(n_input_channels=3, n_output_channels=3)

    D = NLayerDiscriminator(3, n_layers=3)

    G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    #
    # dataset = Subset(
    #     StyleTransferDataset('dataset/training', transform=ToTensor()),
    #     range(1)
    # )

    dataset = StyleTransferDataset('../dataset/mondrian',
                                   transform=transforms.Compose([
                                       Crop(),
                                       ToTensor()
                                   ]))
    dataloader = DataLoader(dataset, batch_size=1,
                                 shuffle=True, num_workers=4)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Working on device: ', device)

    G.to(device)
    D.to(device)

    # G,
    # D,
    # G_optimizer,
    # D_optimizer,
    # dataset,
    # device,
    # criterion,
    # epochs = 5):

    try:
        train_net(
            G,
            D,
            G_optimizer,
            D_optimizer,
            dataset=dataloader,
            device=device,
            criterion=torch.nn.MSELoss(),
            epochs=1000
        )
    except KeyboardInterrupt:
        torch.save(G.state_dict(), 'G_INTERRUPTED.pth')
        torch.save(D.state_dict(), 'D_INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)