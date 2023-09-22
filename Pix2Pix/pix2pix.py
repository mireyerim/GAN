import os
from PIL import Image
import torch.nn as nn
import torch.utils.data
import torchvision
import numpy as np
from torchvision import transforms, datasets
from torchvision.utils import save_image
from pix2pix_init import D, G

num_epoch = 1000
batch_size = 32
learning_rate = 0.0002
img_size = 256
num_channel = 3

dir_nas = "Y:/user/yrso/result/Facades/generated/"
dir_nas_c = "Y:/user/yrso/result/Facades/condition/"
dir_train = "Y:/user/yrso/data/Facades/train"
dir_test = "Y:/user/yrso/data/Facades/test"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now using {} divices".format(device))


# preprocessing

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

class FacadesDataset():
    def __init__(self, path, transform=False):
        super().__init__()
        self.path_img = os.path.join(path, 'b')
        self.path_cond = os.path.join(path, 'a')
        self.img_name_d = [x for x in os.listdir(self.path_img)]
        self.img_name_c = [x for x in os.listdir(self.path_cond)]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path_img, self.img_name_d[index]))
        cond = Image.open(os.path.join(self.path_cond, self.img_name_c[index]))

        if self.transform:
            img = self.transform(img)
            cond = self.transform(cond)

        return img, cond
    
    def __len__(self):
        return len(self.img_name_d)
    
dataset_train = FacadesDataset(dir_train, transform)
dataset_test = FacadesDataset(dir_test, transform)

data_loader = torch.utils.data.DataLoader(dataset = dataset_train, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = dataset_test, batch_size = batch_size, shuffle = True)

# Prepare model

D = D.to(device)
G = G.to(device)

criterion_gan = nn.BCELoss()
criterion_L1 = nn.L1Loss()

lambda_pixel = 100                    # weight of criterion_L1
patch = (1, 256//2**4, 256//2**4)     # the number of patch

D_opt = torch.optim.Adam(D.parameters(), lr = learning_rate, betas = (0.5, 0.999))
G_opt = torch.optim.Adam(G.parameters(), lr = learning_rate, betas = (0.5, 0.999))
#D_opt = torch.optim.RMSprop(D.parameters(), lr = learning_rate)
#G_opt = torch.optim.RMSprop(G.parameters(), lr = learning_rate)


# Train
G.train()
D.train()

for epoch in range(num_epoch):
    for i, (images, condition) in enumerate(data_loader):

        batch_size = images.size(0)

        #images
        images = images.to(device)
        condition = condition.to(device)

        #patch label
        label_real = torch.ones(batch_size, *patch, requires_grad=False).to(device)
        label_fake = torch.zeros(batch_size, *patch, requires_grad=False).to(device)                

        '''
        Generator
        '''
        G.zero_grad()

        fake_image = G(images)
        output = D(fake_image, condition)

        Gan_loss = criterion_gan(output, label_real)
        #Gan_loss = - torch.mean(output)
        pixel_loss = criterion_L1(fake_image, condition)

        G_loss = Gan_loss + pixel_loss * lambda_pixel

        G_loss.backward()
        G_opt.step()

        '''
        Discriminator
        '''
        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)

        D.zero_grad()

        output = D(condition, images)
        real_loss = criterion_gan(output, label_real)
        #real_loss = - torch.mean(output)

        output = D(fake_image.detach(), images)
        fake_loss = criterion_gan(output, label_fake)
        #fake_loss = torch.mean(output)
        
        D_loss = real_loss + fake_loss
        D_loss.backward()
        D_opt.step()
        

    print("Epoch [{}/{}] D_loss: {:.4f} G_loss: {:.4f}"
          .format(epoch+1, num_epoch, D_loss.item(), G_loss.item()))
    
    #save weights
    path = 'Y:/user/yrso/result/Facades/weight/'
    G_weight = os.path.join(path, 'G_weight.pt')
    D_weight = os.path.join(path, 'D_weight.pt')

    torch.save(G.state_dict(), G_weight)
    torch.save(D.state_dict(), D_weight)


    #save generated images

    G.eval()
    
    with torch.no_grad():
        
        for images, condition in test_loader:
            fake = G(images.to(device)).detach().cpu()
            real = condition.cpu()
            break

        save_image(fake, os.path.join(dir_nas, 'generated{}.png'.format(epoch + 1)))
        save_image(real, os.path.join(dir_nas_c, 'condition{}.png'.format(epoch + 1)))