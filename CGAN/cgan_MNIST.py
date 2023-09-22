import os
import torch.nn as nn
import torch.utils.data
import torchvision
import numpy as np
from torchvision import transforms, datasets
from torchvision.utils import save_image

num_epoch = 200
batch_size = 64
learning_rate = 0.0002
img_size = 28
num_channel = 1

dir_nas = "Y:/user/yrso/result/CGAN_result_MNIST/"
noise_size = 100


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now using {} divices".format(device))


transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
    ])

train_dataset = datasets.MNIST(root ="./data/MNIST",
                               train = True,
                               download = True,
                               transform = transform)

data_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
    )

def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data *= 0.1
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            m.weight.data *= 0.1
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data *= 0.1
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Discriminator(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(img_size * img_size * num_channel + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, labels):
        inputs = torch.flatten(inputs, 1)
        conditional = self.label_embedding(labels)
        conditional_inputs = torch.cat([inputs, conditional], dim=-1)
        out = self.main(conditional_inputs)

        return out

class Generator(nn.Module):
    def __init__(self, num_classes = 10):
        super(Generator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(100 + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, num_channel * img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, inputs, labels):
        conditonal_inputs = torch.cat([inputs, self.label_embedding(labels)], dim=-1)
        out = self.main(conditonal_inputs)
        out = out.reshape(out.size(0), num_channel, img_size, img_size)

        return out 
    
D = Discriminator().to(device)
D.apply(_initialize_weights)
G = Generator().to(device)
G.apply(_initialize_weights)

criterion = nn.BCELoss()

D_opt = torch.optim.Adam(D.parameters(), lr = learning_rate)
G_opt = torch.optim.Adam(G.parameters(), lr = learning_rate)

"""
Train
"""
fixed_noise = torch.randn(batch_size, noise_size).to(device)
fixed_conditional = torch.randint(0, 10, (64,)).to(device)
print(fixed_conditional)

for epoch in range(num_epoch):
    for i, (images, label) in enumerate(data_loader):

        batch_size = images.size(0)

        real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
        fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

        images = images.to(device)
        label = label.to(device)        

        z = torch.randn(batch_size, noise_size).to(device)
        conditional = torch.randint(0, 10, (batch_size,)).to(device)

        '''
        Discriminator
        '''
        D_opt.zero_grad()

        real_output = D(images, label)
        real_loss = criterion(real_output, real_label)
        real_loss.backward()

        fake = G(z, conditional)
        fake_output = D(fake.detach(), conditional)
        fake_loss = criterion(fake_output, fake_label)
        fake_loss.backward()

        D_loss = real_loss + fake_loss
        D_opt.step()


        '''
        Generator
        '''
        G_opt.zero_grad()

        fake_output = D(fake, conditional)
        G_loss = criterion(fake_output, real_label)

        G_loss.backward()
        G_opt.step()


    print("Epoch [{}/{}] D_loss: {:.4f} G_loss: {:.4f}"
          .format(epoch+1, num_epoch, D_loss.item(), G_loss.item()))
    
    samples = G(fixed_noise, fixed_conditional)
    save_image(samples, os.path.join(dir_nas, 'CGAN_fake_samples{}.png'.format(epoch + 1)))