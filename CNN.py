import torch
from torch import nn
import torch.nn.functional as F
from ResBlock import ResBlock

class CNN(nn.Module):
    def __init__(self,window_size):
        super(CNN,self).__init__()
        output_channel_list=[8,16,32,64]
        self.res_block1=ResBlock(input_channel=1,output_channel=output_channel_list[0])
        self.res_block2=ResBlock(input_channel=output_channel_list[0],output_channel=output_channel_list[1])
        self.res_block3=ResBlock(input_channel=output_channel_list[1],output_channel=output_channel_list[2])
        self.res_block4=ResBlock(input_channel=output_channel_list[2],output_channel=output_channel_list[3])
        self.linear=nn.Linear(in_features=window_size*output_channel_list[-1],out_features=1)

    def forward(self,x):
        out=self.res_block1(x)
        out=self.res_block2(out)
        out=self.res_block3(out)
        out=self.res_block4(out)
        out=torch.nn.Flatten(out)
        out=F.relu(self.linear(out))
        return out



