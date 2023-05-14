#!/usr/bin/env python
#-*- encoding: utf-8 -*-

import os
import torch.nn as nn
import torch.nn.functional as F

class Mnist(nn.Module):
    def __init__(self):
        super (Mnist, self).__init__()

        # conv - batch - pool
        def CBRP2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers=[]
            layers+=[
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=bias)]
            layers+=[nn.BatchNorm2d(num_features=out_channels)]
            layers+=[nn.ReLU()]
            layers+=[nn.MaxPool2d(kernel_size=2, stride=2)]
            
            return nn.Sequential(*layers)


        # (n, 28, 28, 1)    
        self.cbrp1 = CBRP2d(in_channels=1, out_channels=5)
        # (n, 14, 14, 5)
        self.cbrp2 = CBRP2d(in_channels=5, out_channels=10)
        # (n, 7,  7,  10)
        self.linear = nn.Linear(7 * 7 * 10, 10, bias=True)
        # (n, 10)
        
        #nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, x):
        out_cbrp_1  = self.cbrp1(x)
        out_cbrp_2  = self.cbrp2(out_cbrp_1)
        out_cbrp_2  = out_cbrp_2.view(out_cbrp_2.size(0), -1)
        out_linear  = self.linear(out_cbrp_2)
        output      = F.log_softmax(out_linear, dim=1)
        return output