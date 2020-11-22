import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class DarkNet19(nn.Module):
    def __init__(self,config):
        super(DarkNet19, self).__init__()
        self.config = config

        if(self.config['c_pretrained_weights_loc']):
            pass
        # ======================= Conv block 1 =======================
        self.conv0 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),stride=1,padding=1)
        self.conv0_bn = nn.BatchNorm2d(32)
        self.mp0 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        # ======================= Conv block 2 =======================
        self.conv1 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        # ======================= Conv block 3 =======================
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(1,1),stride=1,padding=0)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        # ======================= Conv block 4 =======================
        self.conv5 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=(1,1),stride=1,padding=0)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=1)
        self.conv7_bn = nn.BatchNorm2d(256)
        self.mp3 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        # ======================= Conv block 5 =======================
        self.conv8 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=1,padding=1)
        self.conv8_bn = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=(1,1),stride=1,padding=0)
        self.conv9_bn = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=1,padding=1)
        self.conv10_bn = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=(1,1),stride=1,padding=0)
        self.conv11_bn = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=1,padding=1)
        self.conv12_bn = nn.BatchNorm2d(512)
        self.mp4 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        # ======================= Conv block 6 =======================
        self.conv13 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(3,3),stride=1,padding=1)
        self.conv13_bn = nn.BatchNorm2d(1024)
        self.conv14 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=(1,1),stride=1,padding=0)
        self.conv14_bn = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(3,3),stride=1,padding=1)
        self.conv15_bn = nn.BatchNorm2d(1024)
        self.conv16 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=(1,1),stride=1,padding=0)
        self.conv16_bn = nn.BatchNorm2d(512)
        self.conv17 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(3,3),stride=1,padding=1)
        self.conv17_bn = nn.BatchNorm2d(1024)

        # ======================= Conv block 7 =======================
        self.conv18 = nn.Conv2d(in_channels=1024,out_channels=config['num_classes'],kernel_size=(1,1),stride=1,padding=1)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
    
    def open_weights(self):
        weights = np.fromfile("../" + self.config['c_pretrained_weights_loc'])
        print(len(weights))
        pass

    def load_from_npz(self, weights_loc, num_conv=None):
        print("== Loading Pretrained Weights ==")
        from collections import OrderedDict
        def split(delimiters, string, maxsplit=0):
            import re
            regexPattern = '|'.join(map(re.escape, delimiters))
            return re.split(regexPattern, string, maxsplit)

        dest_src = OrderedDict(self.config['darknet19'])        
        params = np.load(weights_loc)
        own_dict = self.state_dict()
        keys = list(own_dict.keys())
        
        layer_num = 0
        for idx,(key,val) in enumerate(dest_src.items()):
            layer_num = int(split([".","_"],key)[0].split('v')[1])
            list_key = key.split('.')
            ptype = dest_src['{}.{}'.format(list_key[-2], list_key[-1])]
            src_key = '{}-convolutional/{}:0'.format(layer_num, ptype)
            param = torch.from_numpy(params[src_key])
            if(key == 'conv18.weight' and ptype == 'kernel'):
                param = param[:,:,:,:4]
                param = param.permute(3, 2, 0, 1)
            elif(ptype == 'kernel'):
                param = param.permute(3, 2, 0, 1)
            if(key == 'conv18.bias'):
                param = param[:4]
            own_dict[key].copy_(param)
        print("== Finished Loading Pretrained Weights ==")
        

    def forward(self,x):
        # ======================= Conv block 1 =======================
        output = F.leaky_relu(self.conv0_bn(self.conv0(x)))
        output = self.mp0(output)
        
        # ======================= Conv block 2 =======================
        output = F.leaky_relu(self.conv1_bn(self.conv1(output)))
        output = self.mp1(output)

        # ======================= Conv block 3 =======================
        output = F.leaky_relu(self.conv2_bn(self.conv2(output)))
        output = F.leaky_relu(self.conv3_bn(self.conv3(output)))
        output = F.leaky_relu(self.conv4_bn(self.conv4(output)))
        output = self.mp2(output)

        # ======================= Conv block 4 =======================
        output = F.leaky_relu(self.conv5_bn(self.conv5(output)))
        output = F.leaky_relu(self.conv6_bn(self.conv6(output)))
        output = F.leaky_relu(self.conv7_bn(self.conv7(output)))
        output = self.mp3(output)

        # ======================= Conv block 5 =======================
        output = F.leaky_relu(self.conv8_bn(self.conv8(output)))
        output = F.leaky_relu(self.conv9_bn(self.conv9(output)))
        output = F.leaky_relu(self.conv10_bn(self.conv10(output)))
        output = F.leaky_relu(self.conv11_bn(self.conv11(output)))
        output = F.leaky_relu(self.conv12_bn(self.conv12(output)))
        conv_block_5 = output
        output = self.mp4(output)
        
        # ======================= Conv block 6 =======================
        output = F.leaky_relu(self.conv13_bn(self.conv13(output)))
        output = F.leaky_relu(self.conv14_bn(self.conv14(output)))
        output = F.leaky_relu(self.conv15_bn(self.conv15(output)))
        output = F.leaky_relu(self.conv16_bn(self.conv16(output)))
        output = F.leaky_relu(self.conv17_bn(self.conv17(output)))
        conv_block_6 = output

        # ======================= Conv block 7 =======================
        # from here the output trains darknet19 for classification
        output = self.conv18(output)
        output = self.gap(output)
        # [B,C,H*W]
        output = output.view(output.shape[0],output.shape[1],output.shape[2]*output.shape[3]).contiguous()
        output = F.softmax(output,dim=1)

        return output,conv_block_5,conv_block_6