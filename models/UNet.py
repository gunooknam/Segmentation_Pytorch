import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.initialize_module import weights_init_kaiming_uniform

# ADE20K dataset -----> 150 label
class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDownBlock, self).__init__()
        self.DoubleConv=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels))
        self.maxpool = nn.MaxPool2d(2)
        # weights_init_kaiming_uniform(self)

    def forward(self, x):
        out = self.DoubleConv(x)
        out_pool = self.maxpool(out)
        return out, out_pool # residual connection and pool

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, concat_channels,  out_channels, deconv):
        super(UNetUpBlock, self).__init__()
        self.deconv=deconv
        self.in_channels=in_channels
        # deconv vs interpolation
        if self.deconv == True:
            self.upConv=nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2))
        else :
            self.upConv = nn.Sequential( nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False), nn.Conv2d(in_channels, in_channels, kernel_size=1))
        self.doubleCatConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels))

        # weights_init_kaiming_uniform(self)
    def forward(self, concat_tensor, x):
        x = self.upConv(x)
        dH = concat_tensor.size()[2] - x.size()[2]
        dW = concat_tensor.size()[3] - x.size()[3]
         # (H,W) ---->   left,    right,      up,      ,down  ===> padding
        x = F.pad(x, (dW // 2, dW - dW // 2, dH // 2, dH - dH // 2))
        x = torch.cat((concat_tensor, x), dim=1)
        x = self.doubleCatConv(x)
        return x

class UNet(nn.Module):
    def __init__(self, deconv=True):
        super(UNet, self).__init__()
        self.down_block_1 = UNetDownBlock(3, 64)
        self.down_block_2 = UNetDownBlock(64, 128)
        self.down_block_3 = UNetDownBlock(128, 256)
        self.down_block_4 = UNetDownBlock(256, 512)
        self.down_block_5 = UNetDownBlock(512, 1024)
        self.up_block_1 = UNetUpBlock(512 + 512, 512, 512, deconv)
        self.up_block_2 = UNetUpBlock(256 + 256, 256, 256, deconv)
        self.up_block_3 = UNetUpBlock(128 + 128, 128, 128, deconv)
        self.up_block_4 = UNetUpBlock(64 + 64, 64, 64, deconv)
        self.Conv = nn.Conv2d(64, 150, kernel_size=1, padding=0) # 150 label !
        weights_init_kaiming_uniform(self)

    def forward(self, x):
        print(x.size())
        enc_1, pool_1 = self.down_block_1(x)
        enc_2, pool_2 = self.down_block_2(pool_1)
        enc_3, pool_3 = self.down_block_3(pool_2)
        enc_4, pool_4 = self.down_block_4(pool_3)
        enc_5, pool_5 = self.down_block_5(pool_4)
        dec_1 = self.up_block_1(enc_4, enc_5)
        dec_2 = self.up_block_2(enc_3, dec_1)
        dec_3 = self.up_block_3(enc_2, dec_2)
        dec_4 = self.up_block_4(enc_1, dec_3)
        return self.Conv(dec_4)


if __name__ =='__main__':
    a=torch.rand(1,3,574,574) #
    model = UNet()
    #print(model(a).size())


