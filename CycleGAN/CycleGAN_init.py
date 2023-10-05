import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# generator
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, input):
        return input + self.conv_block(input)

    
class Generator(nn.Module):
    def __init__(self, input, out, n_res=9):
        super(Generator, self).__init__()

        #initial convolution block
        layers = [
            #initial conv block
            nn.ReflectionPad2d(3),
            nn.Conv2d(input, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            #Downsampling
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]

        #Residual Block
        for _ in range(n_res):
            layers.append(ResidualBlock(256))

        #Upsampling
        layers += [            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        #output layer
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
'''
# generator
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, input):
        input = self.down(input)

        return input
    
class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()   
                  ]
        
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self, input, skip):
        input = self.up(input)
        input = torch.cat((input, skip), 1)

        return input

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.d1 = UnetDown(in_channels, 64, normalize=False)
        self.d2 = UnetDown(64, 128)
        self.d3 = UnetDown(128, 256)
        self.d4 = UnetDown(256, 512, dropout=0.5)
        self.d5 = UnetDown(512, 512, dropout=0.5)
        self.d6 = UnetDown(512, 512, dropout=0.5)
        self.d7 = UnetDown(512, 512, dropout=0.5)
        self.d8 = UnetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UnetUp(512, 512, dropout=0.5)
        self.up2 = UnetUp(1024, 512, dropout=0.5)
        self.up3 = UnetUp(1024, 512, dropout=0.5)
        self.up4 = UnetUp(1024, 512, dropout=0.5)
        self.up5 = UnetUp(1024, 256)
        self.up6 = UnetUp(512, 128)
        self.up7 = UnetUp(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.up1(d8,d7)
        u2 = self.up2(u1,d6)
        u3 = self.up3(u2,d5)
        u4 = self.up4(u3,d4)
        u5 = self.up5(u4,d3)
        u6 = self.up6(u5,d2)
        u7 = self.up7(u6,d1)
        u8 = self.up8(u7)

        return u8
'''

# discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),  #256*256*3 --> 128*128*64
            nn.LeakyReLU(0.2, inplace=True), 

            nn.Conv2d(64, 128, 4, 2, 1),   #64*64*128
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),   #32*32*256
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, padding=1),   #31*31*512
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            #FCN classification layer
            nn.Conv2d(512, 1, 4, padding=1)   #30*30*1 
        )

        self.model = nn.Sequential(*model)

    def forward(self, input):
        input = self.model(input)

        return F.avg_pool2d(input, input.size()[2:]).view(input.size()[0], -1)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))