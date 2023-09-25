from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms, datasets
from torchvision.utils import save_image

import torch.autograd
torch.autograd.set_detect_anomaly(True)

dir_nas = "/home/NAS_mount2/yrso/DCGAN_result_CIFAR/"

manualSeed = random.randint(1, 10000)
print(manualSeed)
random.seed(100)
torch.manual_seed(100)

dataset = datasets.CIFAR10(root="./CIFAR-10", download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(64),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                         shuffle=True, num_workers=2)

cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

channel_n = 3
gpu_n = 1
noise_size = 100
g_filter_n = 64
d_filter_n = 64
epoch_num = 1000

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
            nn.ConvTranspose2d(noise_size, g_filter_n * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_filter_n * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_filter_n * 8, g_filter_n * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_filter_n * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_filter_n * 4, g_filter_n * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_filter_n * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_filter_n * 2, g_filter_n, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_filter_n),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_filter_n, channel_n, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            return output

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(channel_n, d_filter_n, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_filter_n, d_filter_n * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_filter_n * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_filter_n * 2, d_filter_n * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_filter_n * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_filter_n * 4, d_filter_n * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_filter_n * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_filter_n * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
    
G = Generator(gpu_n).to(device)
G.apply(weights_init)

D = Discriminator(gpu_n).to(device)
D.apply(weights_init)

criterion = nn.MSELoss()

D_opt = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
G_opt = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))


fixed_noise = torch.randn(128, noise_size, 1, 1, device=device)


for epoch in range(epoch_num):
    for i, data in enumerate(dataloader):
        '''
        Discriminator
        '''
        D.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), 1, device=device, dtype=torch.float)

        output = D(real_cpu)
        real_loss = 0.5 * torch.mean((output - label)**2)
        real_loss.backward()

        z = torch.randn(batch_size, noise_size, 1, 1, device=device)
        fake = G(z)
        label.fill_(0)

        output = D(fake.detach())
        fake_loss = 0.5 * torch.mean((output - label)**2)
        fake_loss.backward()

        D_loss = real_loss + fake_loss
        D_opt.step()

        '''
        Generator
        '''
        if i % 3 == 0:
            G.zero_grad()
            z = torch.randn(batch_size, noise_size, 1, 1, device=device)
            fake = G(z)
            label.fill_(1)

            output = D(fake)
            G_loss = 0.5 * torch.mean((output - label)**2)
            G_loss.backward()
            G_opt.step()
        
    
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch+1, epoch_num, D_loss.item(), G_loss.item()))    

    fake = G(fixed_noise)
    save_image(fake.detach(), os.path.join(dir_nas, 'DCGAN_fake_samples{}.png'.format(epoch + 1)))