# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import math
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import  StepLR
from barbar import Bar
import torch.nn.functional as F
import PIL.Image as Image
import os
import pandas as pd
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from torch.nn.parameter import Parameter
import random




device='cuda:0' if torch.cuda.is_available() else 'cpu'



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
     

# 
SEED = 40  # 
set_seed(SEED)



def seed_worker(worker_id):
    worker_seed = SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)


data_folder='./data/data'
label_folder='./data/label'
files=os.listdir(data_folder)  

valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'} 
labels=[]
datas=[]
for i in range(len(files)):
    data_name=files[i]
    #os.path.join(A,B)：
    data_path=os.path.join(data_folder,data_name)
    # .replace(A.B)：
    label_name=data_name.replace('data', 'label')  
    label_path=os.path.join(label_folder,label_name)

    if not os.path.isfile(data_path) or not os.path.isfile(label_path):
        continue
    if os.path.splitext(data_name)[1].lower() not in valid_extensions:
        continue
    
    # label=np.array(Image.open(label_path).convert("RGB")).transpose(2,0,1)
    # data=np.array(Image.open(data_path).convert("RGB")).transpose(2,0,1)
    #PIL.Image.open(path_img).convert("RGB")
    
    # label=np.array(Image.open(label_path).convert("L"))
    # data=np.array(Image.open(data_path).convert("L"))
    label=np.array(Image.open(label_path))
    data=np.array(Image.open(data_path))
    label=np.array([label]).astype(np.float32())   #
    data=np.array([data]).astype(np.float32())
    
    #https://www.cnblogs.com/xyzluck/p/12807153.html
    labels.append(torch.from_numpy(label))  
    datas.append(torch.from_numpy(data))


    
class REDNet30(nn.Module):
    def __init__(self, num_layers, num_features):
        super(REDNet30, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features*2, num_features, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)
        return x
class Morlet_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):
        super(Morlet_fast, self).__init__()

        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        # self.kernel_size = kernel_size - 1
        self.kernel_size = kernel_size - 1

        # if kernel_size % 2 == 0:
        #     self.kernel_size = self.kernel_size + 1


        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels))
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels))

    def Morlet(self, p):
        C = pow(pi, 0.25)
        # p = 0.03 * p
        y = C * torch.exp(-torch.pow(p, 2) / 2) * torch.cos(2 * pi * p)  # 创建wavelet的部分
        return y

    def forward(self, waveforms):
        # time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
        #                                  steps=int((self.kernel_size / 2)))  #
        device = self.b_.device 
        # time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
        #                                 steps=int((self.kernel_size / 2)))   # 

        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         16).to(device)  # 

        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        16).to(device)  # 

        # p1 = time_disc_right.cuda() - self.b_.cuda() / self.a_.cuda()
        # p2 = time_disc_left.cuda() - self.b_.cuda() / self.a_.cuda()

        p1 = (time_disc_right - self.b_) / self.a_
        p2 = (time_disc_left - self.b_) / self.a_

        Morlet_right = self.Morlet(p1)
        Morlet_left = self.Morlet(p2)

        # Morlet_filter = torch.cat([Morlet_left, Morlet_right], dim=1)  # 40x1x250
        Morlet_filter = torch.cat([Morlet_left, Morlet_right])

        # self.filters = Morlet_filter.view(self.out_channels, 1, self.kernel_size).cuda()

        # self.filters = Morlet_filter.view(self.out_channels, 1,self.kernel_size,1).cuda()
        self.filters = Morlet_filter.view(self.out_channels, 1, 2, 1).to(device)

        # self.filters =torch.randn(1,1,256,256).cuda()
        # return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)
        return F.conv2d(waveforms, self.filters, stride=1, padding=2, dilation=1, bias=None, groups=1)
        

class CustomDataset(Dataset):#
    #
    def __init__(self,data,label):  
        # TODO
        # 1. Initialize file path or list of file names.
        self.data=data
        self.label=label
        #self.CenterCrop=CenterCrop
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        
        return self.data[index].to(device),self.label[index].to(device)
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.data)

#
#
train_data=CustomDataset(datas[:3200],labels[:3200])
#
test_data=CustomDataset(datas[3200:],labels[3200:])


train_loader = DataLoader(
    dataset = train_data,
    #batch_size = 2,
    batch_size = 4,
    # batch_size = 2,
    # batch_size = 64,
    shuffle = True,
#    num_workers = 1,
     worker_init_fn=seed_worker,  
    generator=torch.Generator().manual_seed(SEED)
    )
test_loader = DataLoader(
    dataset = test_data,
    #batch_size = len(test_data),
    batch_size = 3,
    # batch_size = 128,
    shuffle = True,
#    num_workers = 1,
     worker_init_fn=seed_worker,  
    generator=torch.Generator().manual_seed(SEED)
    )

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        # group into subfeatures
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape
        # print(f"Input shape: {x.shape}") 
        # print(f"Channels: {c}, Groups: {self.groups}")  

        # if c % self.groups != 0:
        #     
        #     x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
        #     c += 1
        #     # raise ValueError(f"Channel number {c} must be divisible by 2")
        # if c // self.groups == 0:
        #     raise ValueError(f"Channel number {c} divided by groups {self.groups} results in zero channels")


        x = x.reshape(b * self.groups, -1, h, w)
        # x = x.reshape(b * self.groups, c // self.groups, h, w)


        if (c // self.groups) % 2 != 0:
            x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
            c += 1

        # channel split into
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


class WACTSNet(nn.Module):
    def __init__(self, pad=1, groups=1):
        super(WACTSNet, self).__init__()
        # Encoder
        self.Encoder = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            
            # nn.Conv2d(1, 16, 5, 1, (2, 2), padding_mode='replicate', groups=groups),
            # in size 256*256
            Morlet_fast(16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # out size 128*128
            nn.BatchNorm2d(16),
            sa_layer(16, 8),  # channel=16，group=16

            
            nn.Conv2d(16, 32, 5, 1, (2, 2), padding_mode='replicate', groups=groups),  # in size 128*128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # out size 64*64
            nn.BatchNorm2d(32),
            sa_layer(32, 16),

            
            nn.Conv2d(32, 40, 5, 1, (2, 2), padding_mode='replicate', groups=groups),  # in size 64*64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # out size 32*32
            nn.BatchNorm2d(40),
            sa_layer(40, 20),

            nn.Conv2d(40, 40, 5, 1, (2, 2), padding_mode='replicate', groups=groups),  # in size 64*64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # out size 32*32
            nn.BatchNorm2d(40)
        )
        self.energy_aggregation = REDNet30(num_layers=15, num_features=40)
        # decoder
        self.Decoder = nn.Sequential(
           
            
            nn.ConvTranspose2d(in_channels=40,
                               out_channels=32,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               output_padding=1, groups=groups),  # 进行输出尺寸修正
            nn.ReLU(),
            nn.BatchNorm2d(32),
            sa_layer(32, 16),
            
            nn.ConvTranspose2d(32, 16, 5, 2, 2, 1, groups=groups),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            sa_layer(16, 8),

            
            nn.ConvTranspose2d(16, 2, 5, 2, 2, 1, groups=groups),
            nn.ReLU(),
            nn.BatchNorm2d(2),
            sa_layer(2, 1),

            nn.ConvTranspose2d(2, 1, 5, 2, 2, 1, groups=groups),
            nn.ReLU(),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        # Encoder
        h = self.Encoder(x)  # [B, 40, 32, 32]

        
        h = self.energy_aggregation(h)  # [B, 40, 32, 32]
        #h = self.channel_adjust(h) 

        # Decoder
        im = self.Decoder(h)
        return h, im
# 网络
WAmodel = WACTSNet().to(device=device )

#优化器
#lr:学习率
optimizer = torch.optim.Adam(WAmodel.parameters(), lr=0.001)
#损失函数
scheduler=StepLR(optimizer, 200, gamma=0.2, last_epoch=-1)
loss_func = nn.MSELoss()

PATH = "./check_WA/"

# Resume = True
# if Resume:
# 	path_checkpoint = '/home/htu/workspace/gx/work2/Crossterm_paper_code/SA_CAE/checkpoint/No.500SA_CAE.pth'
# 	checkpoint = torch.load(path_checkpoint, map_location = torch.device('cpu'))
# 	CAEmodel.load_state_dict(checkpoint)



train_losses = []
eval_losses = []
a_values = []  
b_values = []
#
for epoch in range(800):
    # train_loss_epoch, val_loss_epoch = 0, 0
    train_loss_epoch = 0
    print('epoch',epoch)
   
    for b_x, b_y in Bar(train_loader):
        WAmodel.train()

        _,output = WAmodel(b_x)
        
        loss = loss_func(output, b_y)
        
        optimizer.zero_grad()
        #loss.backward()：
        loss.backward()
        #optimizer.step()：
        optimizer.step()
        train_loss_epoch += loss.item() * b_x.size(0)
        # train_num += b_x.size(0)

        
    df_a = pd.DataFrame({
        'Epoch': np.arange(len(a_values)),
        'a_value': a_values
    })
    df_a.to_excel("./check_WA/parameter_a.xlsx", index=False)


    df_b = pd.DataFrame({
        'Epoch': np.arange(len(b_values)),
        'b_value': b_values
    })
    df_b.to_excel("./check_WA/parameter_b.xlsx", index=False)


    df_loss = pd.DataFrame({
        'Epoch': np.arange(len(train_losses)),
        'Train_Loss': train_losses,
        'Valid_Loss': eval_losses
    })
    df_loss.to_excel("./check_WA/loss_values.xlsx", index=False)
    morlet_layer = WAmodel.Encoder[0]
    a_current = morlet_layer.a_.data[0].cpu().numpy()  
    b_current = morlet_layer.b_.data[0].cpu().numpy()  
    a_values.append(a_current)
    b_values.append(b_current)
    train_losses.append(train_loss_epoch / len(train_data))
    scheduler.step()
    if (epoch+1)%10 == 0:
        torch.save(WAmodel.state_dict(),(PATH+f'No.{epoch+1}WA.pth'))
    print('loss',train_loss_epoch)


    CAEmodel=WAmodel.to(device='cuda:0')
    #这里测试ae
    val_loss_epoch = 0
    val_num = 0
    outs=[]
    inputs=[]
   
    for step, (b_x, b_y) in enumerate(test_loader):

            
           WAmodel.eval()
          
           _, output= WAmodel(b_x)
           loss = loss_func(output, b_y)
           val_loss_epoch += loss.item() * b_x.size(0)
           val_num += b_x.size(0)
           outs.append(output.cpu().detach().numpy())
           inputs.append(b_x.cpu().numpy())
            # print(val_num)

    eval_losses.append(val_loss_epoch / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'
            .format(epoch, train_loss_epoch / len(train_data),
                    val_loss_epoch / len(test_data)))

plt.plot(np.arange(len(train_losses)), train_losses,label="train loss")


plt.plot(np.arange(len(eval_losses)), eval_losses, label="valid loss")

plt.legend() 
plt.xlabel('epoches')
#plt.ylabel("epoch")
plt.title('Model loss')
plt.show()
plt.savefig("./check_WA/WA_energy_loss_pic.png")


plt.figure(figsize=(12, 8))


plt.subplot(2, 1, 1)
plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
plt.plot(np.arange(len(eval_losses)), eval_losses, label="valid loss")
plt.legend()
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')


plt.subplot(2, 2, 3)
plt.plot(np.arange(len(a_values)), a_values, label='a parameter')
plt.xlabel('epochs')
plt.ylabel('Value')
plt.title('Morlet Layer Parameter a (1st channel)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(np.arange(len(b_values)), b_values, label='b parameter')
plt.xlabel('epochs')
plt.ylabel('Value')
plt.title('Morlet Layer Parameter b (1st channel)')
plt.legend()

plt.tight_layout()


os.makedirs("./check_WA/parameter_plots", exist_ok=True)
plt.savefig("./check_WA/parameter_plots/loss_and_parameters.png")
plt.close()




idx=0
for j in range(len(outs)):

    for i in range(outs[j].shape[0]):
        
        img1=Image.fromarray(outs[j][i,0].astype('uint8'))
        img2=Image.fromarray(inputs[j][i,0].astype('uint8'))
        img1.save("./reseult_WA/"+'out_'+str(idx)+'.jpg')
        img2.save("./reseult_WA/"+'input_'+str(idx)+'.jpg')

        idx=idx+1







