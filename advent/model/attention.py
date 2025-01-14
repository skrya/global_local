import torch
import numpy as np
import math
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding


class PAM_Module(nn.Module) :
    def __init__(self, in_dim) :
        super(PAM_Module, self).__init__()
        print(f' Hi This is PAM')
        self.channel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8,kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim = -1)
    def forward(self,x) :
        """
         Inputs
            x  = input feature maps (B X C X H X W)
        
        Outputs 
            out = attention_value + input_feature
            attention : B X (H X W) X (H X W)
        """

        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, -1, width*height).permute(0,2,1)
        #proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
        proj_key = x.view(m_batchsize, -1, width*height)
        #proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        #proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        proj_value = x.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class Non_Local_Module(nn.Module) :
    def __init__(self, in_dim) :
        super(Non_Local_Module, self).__init__()
        print(f' Hi This is Non Local')
        self.channel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim*2,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim*2,kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,kernel_size=1)
        #self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim = -1)
    def forward(self,x) :
        """
         Inputs
            x  = input feature maps (B X C X H X W)
        
        Outputs 
            out = attention_value + input_feature
            attention : B X (H X W) X (H X W)
        """

        m_batchsize, C, height, width = x.size()
        #proj_query = x.view(m_batchsize, -1, width*height).permute(0,2,1)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
        #proj_key = x.view(m_batchsize, -1, width*height)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        attention = self.softmax(torch.matmul(proj_query, proj_key))
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        #proj_value = x.view(m_batchsize, -1, width*height)

        out = torch.matmul(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, height, width)

        out = out + x
        return out

class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        print(f' Hi This is CAM. ')
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
        