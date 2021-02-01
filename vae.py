#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image

"""
Lightning에서 몇개의 메소드는 mandatory이다.

- training_step()
- train_dataloader()
-configure_optimizer()

"""



"""
Lightning의 특징으로는 또한
optimizer.zero_grad()
loss.backward()
optimizer.step() 
을 해주지 않아도 된다. 자동적으로 해준다.

또한 .cuda() 같이 GPU로 할지 CPU로 할지 정해주지 않아도 된다.
물론 명시적으로 적을 수 있는 방법이 있기는 한데 그런 사항이 아니면
자동적으로 알아서 학습이 빠른쪽으로 해준다.
하지만 이건 pl.LightningModule 안에 데이터 로더들이 셋되어있을 떄만이다.
만약 main function으로 데이터를 모델에 넣어준다면 당연히 데이터를 cuda나 cpu에 넣어주어야 한다.
"""

class VAE(pl.LightningModule):


    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784,400)
        self.fc21 = nn.Linear(400,20)
        self.fc22 = nn.Linear(400,20)
        self.fc3 = nn.Linear(20,400)
        self.fc4 = nn.Linear(400,784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def decode(self, x):
        h3 = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(h3))


    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z=self.reparameterize(mu,logvar)
        return self.decode(z), mu, logvar



    def training_step(self, batch, batch_idx):
        x,_=batch
        recon_batch, mu, logvar = self(x)        
        loss = self.loss_function(recon_batch, x, mu, logvar)

        return {'loss':loss}

    def save_image(self, data, filename):
        img = data.clone().clamp(0, 255).numpy()
        img = img[0].transpose(1,2,0)
        img= Image.fromarray(img, mode = 'RGB')
        img.save(filename)

    def validation_step(self, batch, batch_idx):
        x, _ =batch
        recon_batch, mu, logvar = self(x)
        val_loss = self.loss_function(recon_batch, x, mu, logvar).item()

        if batch_idx == 0 :
            n = min(x.size(0), 8)
            comparison = torch.cat([x[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])

            self.save_image(comparison.cpu(), './result/reconstruction_'+str(self.current_epoch) + '.png')


        return {'val_loss':val_loss}


    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1,784), reduction='sum')

        KLD = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())

        return BCE+KLD
        

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = 1e-3) 

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train = True, download = True, transform=transforms.ToTensor()),
            batch_size=32, shuffle=True)

        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
            batch_size=32)
        return val_loader

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=32)
    args = parser.parse_args()
    vae = VAE()
    #fast_dev_run 은 컴파일용 이거는 train과 val을 한번씩 만 돈 다음에 잘 돌아가는 지 확인하는것
    
    trainer = pl.Trainer(fast_dev_run = True)
    trainer.fit(vae)
