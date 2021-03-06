from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import pickle


import pandas
import matplotlib.pyplot as plt

# des_dir = r"D:\mbti\ESFP\m"  #저장팔 폴더 위치 
des_dir = "../mbti/"

imageSize = 128    
batchSize = 64   
# batchSize = 64   

dataset = dset.ImageFolder(root=des_dir,
                           transform=transforms.Compose([ # 전처리 작업 
                               transforms.Scale(imageSize), # 이미지 크기 64로 조정 
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 컬러 값이라 채널 3개를 사용  
                               # 이미지의 경우 픽셀 값 하나는 0~255의 값 
                               # ToTensor()로 타입 변경시 0 ~ 1 사이의 값으로 바뀜
                               # Normalize -> -1 ~ 1사이의 값으로 normalized 시킴
                           ]))


# DataLoader 를 통해 데이터를 배치사이즈로 나누어준다 
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size= batchSize,
                                         shuffle=True)

nz     = 100      # dimension of noise vector
nc     = 3        # number of channel - RGB (채널 수 )
ngf    = 128       # generator 필터 조정 
ndf    = 128       # discriminator 필터 조정
niter  = 500     # epoch
lr     = 0.0001   # learning rate
beta1  = 0.5      # hyper parameter of Adam optimizer
ngpu   = 1        # number of using GPU

imageSize = 128   # 만들어지는 이미지의 크기 
batchSize = 64   # 미니배치의 크기 
# batchSize = 64   

outf = "../Tensorflow/_mbti/output/"
# D:\Tensorflow\_mbti

def weights_init(m): # 너무 랜덤한 가중치를 주지 않기 위해서 지정. 
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02) # 평균 0, 표준편차 0.02

    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02) # 평균 1.0, 표준편차 0.02
        m.bias.data.fill_(0)


class _netG(nn.Module): # Generator -> 클래스 형태의 모델은 항상 nn.Module을 상속받아야 한다. 

    def __init__(self, ngpu):
        super(_netG, self).__init__() # nn.Module.__init__() 을 실행
        self.ngpu = ngpu
        self.main = nn.Sequential(

            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), 
            # ConvTranspose2d(a, b, c, d, e): a는 input 채널의 수, b는 만들어지는 결과값의 채널의 수 
            # c는 커널의 크기, 즉 Convolution 연산을 수행하는 필터의 크기
            # d는 stride, e는 padding

            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh() # 값을 -1~1사이로

            # state size. (nc) x 64 x 64
        )
    def forward(self, input):
        # 모델이 학습데이터를 입력받아서 순전파를 진행시키는 함수
        # 함수 이름이 반드시 forward여야 한다. 
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module): # Discriminator
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # 이미지 채널 
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() # (0~1사이의 값)으로 확률 구하기 

            # state size. 1
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

# weights 적용 
netG = _netG(ngpu)
netG.apply(weights_init)
print(netG)

netD = _netD(ngpu)
netD.apply(weights_init)
print(netD)

criterion = nn.BCELoss() 
# Binary Cross Entropy를 쓰는 이유 : 
# GAN의 판별자 D는 real or fake를 판단하기 때문에. real일 때 y = 1, fake일 때 y = 0

input = torch.FloatTensor(batchSize, 3, imageSize,imageSize) # 높이, 너비 채널이 아닌 채널, 
noise = torch.FloatTensor(batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)

label = torch.FloatTensor(batchSize)
real_label = 1
fake_label = 0

# GPU 사용 지정 
netD.cuda()
netG.cuda()
criterion.cuda()
input, label = input.cuda(), label.cuda()
noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999)) # 판별자 optimizer
# betas는 기울기와 그 제곱의 실행 평균을 계산하는 데 사용되는 계수
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999)) # 생성자 optimizer

result_dict = {}
loss_D,loss_G,score_D,score_G1,score_G2 = [],[],[],[],[]


for epoch in range(niter):
    for i, data in enumerate(dataloader, 0): # 배치사이즈로 잘린다. 이 때 i는 index, data는 value
        ############################
        # (1) Discriminator Train: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        # train with real
        netD.zero_grad() # 모든 매개변수의 gradient 값을 초기화 시킨다. -> 변화도가 누적되기 때문에 초기화 
        real_cpu, _ = data # datareader는 주소값을 읽어오고 여기서 data에서 실제 데이터를 읽어온다 
        batch_size = real_cpu.size(0)

        real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)

        # input, label 모두 Variable로 wrapping
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv) # labelv에 대한 loss를 구한다 -> lavelv는 진짜인지 아닌ㄴ지 값을 판명한 값 
        errD_real.backward() # loss.backward()
        D_x = output.data.mean() # real data 평균값 

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)

        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label)) 
        # 빈값을 만들었다가 batch size에 따라 0, 또는 1로 채운다 
        # 가짜 라벨로 만들어진다

        output = netD(fake.detach()) 
        # keras에서의 Discriminator.trainable=False와 동일한 의미-> 생성자를 학습시키기 위해 
        
        errD_fake = criterion(output, labelv) 
        errD_fake.backward()
        D_G_z1 = output.data.mean()  # fake data 평균값 

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Generator Train: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  
        # 진짜 라벨로 만들어진다
        
        output = netD(fake) 

        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean() 
        optimizerG.step()


    # 시각화 
    vutils.save_image(real_cpu,
            '%s/real_samples.png' % outf,
            normalize=True)
    fake = netG(fixed_noise)

    vutils.save_image(fake.data,
            '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
            normalize=True)

    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
      % (epoch, niter, i, len(dataloader),
         errD.data, errG.data, D_x, D_G_z1, D_G_z2))

    loss_D.append(errD.data)
    loss_G.append(errG.data)
    score_D.append(D_x)
    score_G1.append(D_G_z1)
    score_G2.append(D_G_z2)
    result_dict = {"loss_D":loss_D,"loss_G":loss_G,"score_D":score_D,"score_G1":score_G1,"score_G2":score_G2}
    
    pickle.dump(result_dict,open("./{}/result_dict.p".format(outf),"wb"))

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG.pth' % (outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (outf))


'''

print 시 나오는 지표 의미
loss_D : Discriminator가 훈련하면서 나오는 loss
loss_G : Generator가 훈련하면서 나오는 loss

gan 식에서
V(D,G)는 GAN의 Loss함수, 목적 함수라고 불리고 D의 목적은 V(D,G)가 최대가 되도록 하는 것.
Discriminator는 가짜 데이터는 0, 진짜 데이터는 1을 출력하게 됨. X는 진짜 데이터를 의미, G(Z)는 가짜 데이터를 의미.
결과적으로 D는 D(x)=1, D(G(z))는 0이 되는 것이 좋은 지표.

D(x) : Discriminator's score -> 1이 될 수록 좋은 지표 
D(G(z)) : Generator's score -> 0이 될수록 좋은 지표 

'''