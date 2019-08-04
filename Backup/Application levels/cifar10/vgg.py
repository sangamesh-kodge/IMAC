#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:45:39 2019

@author: skodge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
        

class vgg(nn.Module):
    def __init__(self,input_size, bit_W, bit_A, sigma, n_classes=10, **kwargs):
        self.bit_A=bit_A
        self.bit_W=bit_W
        self.sigma=sigma
        self.error_initialiser()        
        
        super(vgg, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 64, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.linear1 =nn.Linear(4096,4096,bias=True)
        self.linear2 =nn.Linear(4096,4096,bias=True)
        self.classifier = nn.Linear(4096, n_classes, bias=True)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x,.3)  
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)

        x = F.relu(self.conv3(x))
        x = F.dropout(x,.4)  
        x = F.relu(self.conv4(x))
        x = self.maxpool2(x)        

        x = F.relu(self.conv5(x))
        x = F.dropout(x,.4)  
        x = F.relu(self.conv6(x))
        x = F.dropout(x,.4)  
        x = F.relu(self.conv7(x))
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.dropout(x,.5)    
        x = F.relu(self.linear2(x))
        x = F.dropout(x,.5)    
        x = self.classifier(x)
        return x
    

    
    def Inference(self, x, do_quantise=True, do_add_error=True, mode=False):
        
        x = self.quantise(x,self.bit_A,do_quantise=do_quantise)
        x = self.AddError( self.conv1(x), x.size()[1], self.error_conv1, k=3, st=1, p=1, bit_out=5, mode=mode, do_add_error=do_add_error)
        x = F.relu(x)
        
        x = self.quantise(x,self.bit_A,do_quantise=do_quantise)
        x = self.AddError( self.conv2(x), x.size()[1], self.error_conv2, k=3, st=1, p=1, bit_out=5, mode=mode, do_add_error=do_add_error)
        x = F.relu(x)
        x = self.maxpool1(x)
        
        x = self.quantise(x,self.bit_A,do_quantise=do_quantise)
        x = self.AddError( self.conv3(x), x.size()[1], self.error_conv3, k=3, st=2, p=1,bit_out=5, mode=mode, do_add_error=do_add_error)
        x = F.relu(x)
       
        x = self.quantise(x,self.bit_A,do_quantise=do_quantise)
        x = self.AddError( self.conv4(x), x.size()[1], self.error_conv4, k=3, st=1, p=1,bit_out=5, mode=mode, do_add_error=do_add_error)
        x = F.relu(x)
        x = self.maxpool2(x)
        
        x = self.quantise(x,self.bit_A,do_quantise=do_quantise)
        x = self.AddError( self.conv5(x), x.size()[1], self.error_conv5, k=3, st=1, p=1,bit_out=5, mode=mode, do_add_error=do_add_error)
        x = F.relu(x)
        
        x = self.quantise(x,self.bit_A,do_quantise=do_quantise)
        x = self.AddError( self.conv6(x), x.size()[1], self.error_conv6, k=3, st=2, p=1,bit_out=5, mode=mode, do_add_error=do_add_error)
        x = F.relu(x)
        
        x = self.quantise(x,self.bit_A,do_quantise=do_quantise)
        x = self.AddError( self.conv7(x), x.size()[1], self.error_conv7, k=3, st=1, p=1, bit_out=5, mode=mode, do_add_error=do_add_error)
        x = F.relu(x)
        x = self.maxpool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.quantise(x,self.bit_A,do_quantise=do_quantise)
        x = self.AddError( self.linear1(x), x.size()[1], self.error_linear1, k=1, st=1, p=1, bit_out=5, mode=mode, do_add_error=do_add_error)
        x = F.relu(x)
        
        x = self.quantise(x,self.bit_A,do_quantise=do_quantise)
        x = self.AddError( self.linear2(x), x.size()[1], self.error_linear2, k=1, st=1, p=1, bit_out=5, mode=mode, do_add_error=do_add_error)
        x = F.relu(x)
        
        x = self.quantise(x,self.bit_A,do_quantise=do_quantise)
        x = self.AddError( self.classifier(x), x.size()[1], self.error_classifier, k=1, st=1, p=1, bit_out=5, mode=mode, do_add_error=do_add_error)
        #x = self.classifier(x)
        return x  
    
    
    def quantise(self, x, k, do_quantise=False):
        if( not do_quantise):
            output=x        
        Min=torch.min(x)
        Max=torch.max(x)
        output=torch.round(((2**k)-1)*(x- Min)/(Max-Min))
        output=Min+(Max-Min)*output/((2**k)-1)
        return output
                
    
    def AddError(self, x, inp_channels=0, error_table=0, k=3, st=1, p=1, bit_out=5, mode=True, do_add_error=True):
        if (not do_add_error):
            return x
        noofterms = inp_channels*k*k
        levels = noofterms*(2**bit_out-1)
        Min=x.min()
        Max=x.max()
        if(mode): 
            outp = torch.zeros(x.size()).cuda()
            outn = torch.zeros(x.size()).cuda()
            error_out = torch.round(error_table)
            outp = torch.round(torch.where(x>0,x,outp)*levels/Max)
            outn = torch.round(torch.where(x<0,x,outn)*levels/Min)
            out= (outp+outn)+error_out
            outp=torch.zeros(x.size()).cuda()
            outn=torch.zeros(x.size()).cuda()
            outp=torch.clamp(torch.where(x>0,out,outp),0,levels)*Max/levels
            outn=torch.clamp(torch.where(x<0,out,outn),0,-levels)*Min/levels
            out1=outp+outn
        else: 
            if(k==3):
                outp=torch.zeros(x.size()).cuda()
                outn=torch.zeros(x.size()).cuda()
                n=torch.distributions.normal.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(noofterms)]))
                error_out = n.sample((x.size()[0],x.size()[1],x.size()[2],x.size()[3],)).view(x.size()[0],x.size()[1],x.size()[2],x.size()[3]).cuda()
                outp=torch.round(torch.where(x>0,x,outp)*levels/Max)
                outn=torch.round(torch.where(x<0,x,outn)*levels/Min)
                out= (outp+outn)+error_out
                outp=torch.zeros(x.size()).cuda()
                outn=torch.zeros(x.size()).cuda()
                outp=torch.clamp(torch.where(x>0,out,outp),0,levels)*Max/levels
                outn=torch.clamp(torch.where(x<0,out,outn),0,-levels)*Min/levels
                out1=outp+outn
            else:
                outp=torch.zeros(x.size()).cuda()
                outn=torch.zeros(x.size()).cuda()
                n=torch.distributions.normal.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(noofterms)]))
                error_out = n.sample((x.size()[0],x.size()[1],)).view(x.size()[0],x.size()[1]).cuda()
                outp=torch.round(torch.where(x>0,x,outp)*levels/Max)
                outn=torch.round(torch.where(x<0,x,outn)*levels/Min)
                out= (outp+outn)+error_out
                outp=torch.zeros(x.size()).cuda()
                outn=torch.zeros(x.size()).cuda()
                outp=torch.clamp(torch.where(x>0,out,outp),0,levels)*Max/levels
                outn=torch.clamp(torch.where(x<0,out,outn),0,-levels)*Min/levels
                out1=outp+outn
        return out1
        
    
    def quantise_weigths(self,do_quantise=True):
        self.conv1.weight.data = self.quantise(self.conv1.weight.data,self.bit_W,do_quantise=do_quantise)
        self.conv2.weight.data = self.quantise(self.conv2.weight.data,self.bit_W,do_quantise=do_quantise)
        self.conv3.weight.data = self.quantise(self.conv3.weight.data,self.bit_W,do_quantise=do_quantise)
        self.conv4.weight.data = self.quantise(self.conv4.weight.data,self.bit_W,do_quantise=do_quantise)
        self.conv5.weight.data = self.quantise(self.conv5.weight.data,self.bit_W,do_quantise=do_quantise)
        self.conv6.weight.data = self.quantise(self.conv6.weight.data,self.bit_W,do_quantise=do_quantise)
        self.conv7.weight.data = self.quantise(self.conv7.weight.data,self.bit_W,do_quantise=do_quantise)
        self.linear1.weight.data = self.quantise(self.linear1.weight.data,self.bit_W,do_quantise=do_quantise)
        self.linear2.weight.data = self.quantise(self.linear2.weight.data,self.bit_W,do_quantise=do_quantise)
        self.classifier.weight.data = self.quantise(self.classifier.weight.data,self.bit_W,do_quantise=do_quantise)
        
    def error_initialiser(self) :
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(3*3*3)]))
        self.error_conv1=n.sample((64,32,32,)).view(64,32,32).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(64*3*3)]))
        self.error_conv2=n.sample((64,32,32,)).view(64,32,32).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(64*3*3)]))
        self.error_conv3=n.sample((128,16,16,)).view(128,16,16).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(128*3*3)]))
        self.error_conv4=n.sample((128,16,16,)).view(128,16,16).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(128*3*3)]))
        self.error_conv5=n.sample((256,8,8,)).view(256,8,8).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(256*3*3)]))
        self.error_conv6=n.sample((256,8,8,)).view(256,8,8).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(256*3*3)]))
        self.error_conv7=n.sample((256,8,8,)).view(256,8,8).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(4096)]))
        self.error_linear1=n.sample((4096,)).view(4096).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(4096)]))
        self.error_linear2=n.sample((4096,)).view(4096).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(10)]))
        self.error_classifier=n.sample((10,)).view(10).cuda()
        

 