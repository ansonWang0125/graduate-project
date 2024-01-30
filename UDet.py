import torch
import torch.nn as nn
import torch.nn.functional as F
from bifpn import BiFPN

class UDet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False, batch_norm=False, up_mode='upconv'):
        super(UDet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding

        self.conv1 = UNetConvBlock(in_channels, 64, padding, batch_norm)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = UNetConvBlock(64, 128, padding, batch_norm)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = UNetConvBlock(128, 256, padding, batch_norm)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = UNetConvBlock(256, 512, padding, batch_norm)
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = UNetConvBlock(512, 1024, padding, batch_norm)

        self.channels = [64, 128, 256, 512, 1024]
        
        # self.bifpn = BiFPN(self.features, self.channels[0], 1)
        self.bifpn = BiFPN(self.channels, 1)

        self.up6 = UNetUpBlock(1024, 512, up_mode, padding, batch_norm)
        # self.conv6 = UNetConvBlock(513, 512, padding, batch_norm)

        self.up7 = UNetUpBlock(512, 256, up_mode, padding, batch_norm)
        # self.conv7 = UNetConvBlock(257, 256, padding, batch_norm)

        self.up8 = UNetUpBlock(256, 128, up_mode, padding, batch_norm)
        # self.conv8 = UNetConvBlock(129, 128, padding, batch_norm)

        self.up9 = UNetUpBlock(128, 64, up_mode, padding, batch_norm)
        # self.conv9 = UNetConvBlock(65, 64, padding, batch_norm)

        self.conv10 = nn.Conv2d(64, 1, 1)

    
    def forward(self, x):
        blocks = []
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        blocks.append(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        blocks.append(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        blocks.append(conv3)

        conv4 = self.conv4(pool3)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)
        blocks.append(conv4)

        conv5 = self.conv5(pool4)
        
        features = [conv1, conv2, conv3, conv4, conv5]
        bifpn = self.bifpn(features)
        
        conv6 = self.up6(conv5, blocks[-1], bifpn[3])
        
        conv7 = self.up7(conv6, blocks[-2], bifpn[2])
        
        conv8 = self.up8(conv7, blocks[-3], bifpn[1])
        
        conv9 = self.up9(conv8, blocks[-4], bifpn[0])

#         up6 = self.up6(conv5, blocks[-1], bifpn[3])
#         print(up6.size())
#         print(bifpn[3].size())
#         concat6 = torch.cat((up6, bifpn[3]), dim=1)
#         print(concat6.size())
#         conv6 = self.conv6(concat6)

#         up7 = self.up7(conv6, blocks[-2])
#         concat7 = torch.cat((up7, bifpn[2]), dim=1)
#         conv7 = self.conv7(concat7)

#         up8 = self.up8(conv7, blocks[-3])
#         concat8 = torch.cat((up8, bifpn[1]), dim=1)
#         conv8 = self.conv8(concat8)

#         up9 = self.up9(conv8, blocks[-4])
#         concat9 = torch.cat((up9, bifpn[0]), dim=1)
#         conv9 = self.conv9(concat9)

        conv10 = self.conv10(conv9)

        return torch.sigmoid(conv10)
    
class UNetConvBlock(nn.Module): #每一層都會做2次的convolution，kenrel size 都是3
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        # block.append(nn.ReLU())
        block.append(nn.Mish())
        # block.append(nn.LeakyReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        # block.append(nn.ReLU())
        block.append(nn.Mish())
        # block.append(nn.LeakyReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size + 1, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge, features):
        up = self.up(x)
        # print("Start: ")
        # print(up.size())
        # print(features.size())
        # print("Finish--")
        up = torch.cat((up, features), 1)
        # print("Start: ")
        # print(up.size())
        # print(features.size())
        # print("Finish--")
        crop1 = self.center_crop(bridge, up.shape[2:])
        # print("Start: ")
        # print(up.size())
        # print(crop1.size())
        # print("Finish--")
        out = torch.cat([up, crop1], 1)
        # print("Start: ")
        # print(up.size())
        # print(crop1.size())
        # print(out.size())
        # print("Finish--")
        out = self.conv_block(out)

        return out