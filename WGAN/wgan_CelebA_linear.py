from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms, datasets
from torchvision.utils import save_image

dir_nas = "/home/NAS_mount2/yrso/WGAN_result_celebA/"
dir_data = "/home/NAS_mount2/yrso/data/celebA/img_align_celeba"

manualSeed = random.randint(1, 10000)
random.seed(40)
torch.manual_seed(40)

dataset = datasets.ImageFolder(dir_data, transform=transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                         shuffle=True, num_workers=16)

cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

channel_n = 3
gpu_n = 1
noise_size = 100
g_filter_n = 64
d_filter_n = 64
epoch_num = 200

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(noise_size, 128),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, g_filter_n*g_filter_n*channel_n),
            nn.Tanh()
        )

    def forward(self, input):
        img = self.main(input)
        img = img.view(img.shape[0], channel_n, g_filter_n, g_filter_n)
        return img

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(channel_n*d_filter_n*d_filter_n, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1)
        )

    def forward(self, input):
        input = input.view(input.shape[0], -1)
        return self.main(input)

G = Generator(gpu_n).to(device)
#G.apply(weights_init)

D = Discriminator(gpu_n).to(device)
#D.apply(weights_init)

#criterion = nn.MSELoss()

#D_opt = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
#G_opt = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_opt = torch.optim.RMSprop(D.parameters(), lr=0.0002)
G_opt = torch.optim.RMSprop(G.parameters(), lr=0.0002)

fixed_noise = torch.randn(64, noise_size).to(device)

print('Ready')
for epoch in range(epoch_num):
    for i, data in enumerate(dataloader):
        '''
        Discriminator
        '''
        D_opt.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        #label = torch.full((batch_size,), 1, device=device, dtype=torch.float)

        #output = D(real_cpu)
        #real_loss = 0.5 * torch.mean((output - label)**2)
        #real_loss.backward()

        z = torch.randn(batch_size, noise_size).to(device)
        fake = G(z)
        #label.fill_(0)

        #output = D(fake.detach())
        #fake_loss = 0.5 * torch.mean((output - label)**2)
        #fake_loss.backward()

        #D_loss = real_loss + fake_loss
        D_loss = - torch.mean(D(real_cpu)) + torch.mean(D(fake.detach()))
        D_loss.backward()
        D_opt.step()

        #for p in D.parameters():
            #p.data.clamp_(-0.01, 0.01)

        '''
        Generator
        '''
        G_opt.zero_grad()
        fake = G(z)
        #label.fill_(1)

        #output = D(fake)
        G_loss = -torch.mean(D(fake))
        G_loss.backward()
        G_opt.step()

    
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch+1, epoch_num, D_loss.item(), G_loss.item()))    

    fake = G(fixed_noise)
    save_image(fake.detach(), os.path.join(dir_nas, 'WGAN_fake_samples{}.png'.format(epoch + 1)))