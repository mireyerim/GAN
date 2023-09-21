import os
import torch.nn as nn
import torch.utils.data
import torchvision
import numpy as np
from torchvision import transforms, datasets
from torchvision.utils import save_image

num_epoch = 1000
batch_size = 64
learning_rate = 0.0002
img_size = 32
num_channel = 3

dir_nas = "/home/NAS_mount2/yrso/"
noise_size = 100
hidden_size1 = 256
hidden_size2 = 512
hidden_size3 = 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now using {} divices".format(device))


transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

train_dataset = datasets.CIFAR10(root ="./data/CIFAR-10",
                               train = True,
                               download = True,
                               transform = transform)

data_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
    )

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(img_size*img_size*num_channel, hidden_size3)
        self.linear2 = nn.Linear(hidden_size3, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size1)
        self.linear4 = nn.Linear(hidden_size1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(noise_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, img_size*img_size*num_channel)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        x = self.tanh(x)
        return x
    
D = Discriminator()
G = Generator()

D = D.to(device)
G = G.to(device)

criterion = nn.BCELoss()

D_opt = torch.optim.Adam(D.parameters(), lr = learning_rate)
G_opt = torch.optim.Adam(G.parameters(), lr = learning_rate)

"""
Train
"""

for epoch in range(num_epoch):
    for i, (images, label) in enumerate(data_loader):

        real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
        fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

        real_images = images.reshape(batch_size, -1).to(device)

        '''
        Generator
        '''
        D_opt.zero_grad()
        G_opt.zero_grad()

        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = G(z)

        G_loss = criterion(D(fake_images), real_label)

        G_loss.backward()
        G_opt.step()


        '''
        Discriminator
        '''
        D_opt.zero_grad()
        G_opt.zero_grad()

        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = G(z)

        real_loss = criterion(D(real_images), real_label)
        fake_loss = criterion(D(fake_images), fake_label)
        D_loss = (real_loss + fake_loss)/2

        D_loss.backward()
        D_opt.step()

        D_performance = D(real_images).mean()
        G_performance = D(fake_images).mean()

    print("Epoch [{}/{}] D_loss: {:.4f} G_loss: {:.4f}"
          .format(epoch+1, num_epoch, D_loss.item(), G_loss.item()))
    
    if (epoch+1)%10 == 0:
        samples = fake_images.reshape(batch_size, 3, 32, 32)
        save_image(samples, os.path.join(dir_nas, 'GAN_fake_samples{}.png'.format(epoch + 1)))