import torch
import torch.nn as nn

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

class Unet(nn.Module):
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
    

# discriminator
class Discriminator_layer(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, 2, 1)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2))

        self.layer = nn.Sequential(*layers)

    def forward(self, input):
        input = self.layer(input)

        return input
    
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.d1 = Discriminator_layer(in_channels*2, 64, normalize=False)
        self.d2 = Discriminator_layer(64, 128)
        self.d3 = Discriminator_layer(128, 256)
        self.d4 = Discriminator_layer(256, 512)

        self.patch = nn.Conv2d(512, 1, 3, padding=1) # 16 x 16 patch

    def forward(self, input, cond):
        y = torch.cat((input, cond), 1)
        y = self.d1(y)
        y = self.d2(y)
        y = self.d3(y)
        y = self.d4(y)
        y = self.patch(y)
        #y = torch.sigmoid(y)

        return y


def _initialize_weights(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)


# model generation

G = Unet()
G.apply(_initialize_weights)
D = Discriminator()
D.apply(_initialize_weights)