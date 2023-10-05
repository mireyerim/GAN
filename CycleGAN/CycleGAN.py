import os
import itertools
from PIL import Image
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
from CycleGAN_init import *

num_epoch = 1000
batch_size = 4
learning_rate = 0.0002
img_size = 256
num_channel = 3

dir_nas = "Y:/user/yrso/result/CycleGAN_result_facade/try5/"
dir_train = "Y:/user/yrso/data/Facades/train"
dir_test = "Y:/user/yrso/data/Facades/test"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now using {} divices".format(device))


# preprocessing

transform_base = transforms.Compose([
    transforms.Resize(int(img_size * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #transforms.GaussianBlur(kernel_size=(25, 25), sigma=(1.0, 2.0)),
    ])

transform_pho = transforms.Compose([
    transforms.Resize(int(img_size * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

transform_t = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

class Dataset():
    def __init__(self, root, transform_base=None, transform_pho=None):
        super().__init__()
        self.root_base = os.path.join(root, 'b')
        self.root_photo = os.path.join(root, 'a')

        self.transform_b = transform_base
        self.transform_p = transform_pho

        self.bases = os.listdir(self.root_base)
        self.photos = os.listdir(self.root_photo)

        self.base_l = len(self.bases)
        self.photo_l = len(self.photos)

    def __getitem__(self, index):
        basen = self.bases[index % self.base_l]
        photon = self.photos[index % self.photo_l]

        base = Image.open(os.path.join(self.root_base, basen))
        photo = Image.open(os.path.join(self.root_photo, photon))

        if self.transform_b and self.transform_p:
            base = self.transform_b(base)
            photo = self.transform_p(photo)

        return base, photo
    
    def __len__(self):
        return max(len(self.bases), len(self.photos))
    

dataset_train = Dataset(dir_train, transform_base=transform_base, transform_pho=transform_pho)
dataset_test = Dataset(dir_test, transform_base=transform_t, transform_pho=transform_t)

data_loader = torch.utils.data.DataLoader(dataset = dataset_train, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = dataset_test, batch_size = 4, shuffle = True)

# Prepare model
D_b = Discriminator().to(device)
D_p = Discriminator().to(device)
G_b2p = Generator(3, 3).to(device)
G_p2b = Generator(3, 3).to(device)

D_b.apply(weights_init_normal)
D_p.apply(weights_init_normal)
G_b2p.apply(weights_init_normal)
G_p2b.apply(weights_init_normal)

criterion_gan = nn.MSELoss()
criterion_L1 = nn.L1Loss()

D_b_opt = torch.optim.Adam(D_b.parameters(), lr = learning_rate, betas = (0.5, 0.999))
D_p_opt = torch.optim.Adam(D_p.parameters(), lr = learning_rate, betas = (0.5, 0.999))
G_opt = torch.optim.Adam(itertools.chain(G_b2p.parameters(), G_p2b.parameters()), lr = learning_rate, betas = (0.5, 0.999))

fake_photo_buffer = ReplayBuffer()
fake_base_buffer = ReplayBuffer()

print('ready')
# Train
for epoch in range(num_epoch):
    for i, (base, photo) in enumerate(data_loader):
        batch_size = base.size(0)

        real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
        fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

        #images
        base = base.to(device)
        photo = photo.to(device)          

        '''
        Generator
        '''
        G_opt.zero_grad()

        #identity loss
        fake_photo = G_b2p(photo)
        fake_base = G_p2b(base)

        loss_p_i = criterion_L1(fake_photo, photo) * 5.0
        loss_b_i = criterion_L1(fake_base, base) * 5.0

        #generator loss
        fake_photo = G_b2p(base)
        fake_photo_r = D_p(fake_photo)
        loss_p_g = criterion_gan(fake_photo_r, real_label)

        fake_base = G_p2b(photo)
        fake_base_r = D_b(fake_base)
        loss_b_g = criterion_gan(fake_base_r, real_label)

        #cycle loss
        regen_base = G_p2b(fake_photo)
        loss_b_c = criterion_L1(regen_base, base) * 10.0

        regen_photo = G_b2p(fake_base)
        loss_p_c = criterion_L1(regen_photo, photo) * 10.0

        G_loss = loss_p_i + loss_p_g + loss_p_c + loss_b_i + loss_b_g + loss_b_c
        G_loss.backward()

        G_opt.step()

        '''
        Discriminator
        '''
        #Monet discriminator
        D_b_opt.zero_grad()

        #real loss
        pred_real = D_b(base)
        loss_real = criterion_gan(pred_real, real_label)

        #fake loss
        fake_base = fake_base_buffer.push_and_pop(fake_base)
        pred_fake = D_b(fake_base.detach())
        loss_fake = criterion_gan(pred_fake, fake_label)

        D_b_loss = (loss_real + loss_fake) * 0.5
        D_b_loss.backward()

        D_b_opt.step()


        #Photo discriminator
        D_p_opt.zero_grad()

        #real loss
        pred_real = D_p(photo)
        loss_real = criterion_gan(pred_real, real_label)

        #fake loss
        fake_photo = fake_photo_buffer.push_and_pop(fake_photo)
        pred_fake = D_p(fake_photo.detach())
        loss_fake = criterion_gan(pred_fake, fake_label)

        D_p_loss = (loss_real + loss_fake) * 0.5
        D_p_loss.backward()

        D_p_opt.step()


    print("Epoch [{}/{}] D_m_loss: {:.4f} D_p_loss: {:.4f} G_loss: {:.4f}"
          .format(epoch+1, num_epoch, D_b_loss.item(), D_p_loss.item(), G_loss.item()))


    #test
    with torch.no_grad():
        
        for b, p in test_loader:
            gen = Image.new('RGB', (1024, 1024))

            b2p_f = G_b2p(b.to(device)).detach().cpu()
            ba = b.cpu()

            p2b_f = G_p2b(p.to(device)).detach().cpu()
            ph = p.cpu()

            for i in range(b.size(0)):
                b2p_i = transforms.ToPILImage()(b2p_f[i].squeeze(0))
                ba_i = transforms.ToPILImage()(ba[i].squeeze(0))

                p2b_i = transforms.ToPILImage()(p2b_f[i].squeeze(0))
                ph_i = transforms.ToPILImage()(ph[i].squeeze(0))
                
                gen.paste(ba_i, (0, i * 256))
                gen.paste(ph_i, (512, i * 256))
                gen.paste(b2p_i, (256, i * 256))
                gen.paste(p2b_i, (768, i * 256))
            break

        gen.save(os.path.join(dir_nas, 'generated{}.png'.format(epoch + 1)))