#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:21:07 2019

@author: skodge
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import sys
        

class lenet(nn.Module):
    def __init__(self,input_size, bit_W, bit_A, sigma, n_classes=10, **kwargs):
        self.bit_A=bit_A
        self.bit_W=bit_W
        self.sigma=sigma
        self.error_initialiser()        
        
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 6, kernel_size=5, padding=2, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 =nn.Linear(400,120,bias=True)
        self.linear2 =nn.Linear(120,84,bias=True)
        self.classifier = nn.Linear(84, n_classes, bias=True)


    def forward(self, x):
        #print (x.shape)
        x = F.relu(self.conv1(x))
        #print (x.shape)
        x = self.maxpool1(x)
        #sys.exit()
        x = F.dropout(x,.2)  
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)        
        x = F.dropout(x,.2)  
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.dropout(x,.2)    
        x = F.relu(self.linear2(x))
        x = F.dropout(x,.2)    
        x = self.classifier(x)
        return x
    

    
    def Inference(self, x, do_quantise=True, do_add_error=True, mode=False):
        
        x = self.quantise(x,self.bit_A,do_quantise=do_quantise)
        x = self.AddError( self.conv1(x), x.size()[1], self.error_conv1, k=5, st=1, p=2, bit_out=5, mode=mode, do_add_error=do_add_error)
        x = F.relu(x)
        x = self.maxpool1(x)
        
        
        x = self.quantise(x,self.bit_A,do_quantise=do_quantise)
        x = self.AddError( self.conv2(x), x.size()[1], self.error_conv2, k=5, st=1, p=0, bit_out=5, mode=mode, do_add_error=do_add_error)
        x = F.relu(x)
        x = self.maxpool2(x)
        
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
                
    
    def AddError(self, x, inp_channels=0, error_table=0, k=5, st=1, p=0, bit_out=5, mode=True, do_add_error=True):
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
            if(k>1):
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
        self.linear1.weight.data = self.quantise(self.linear1.weight.data,self.bit_W,do_quantise=do_quantise)
        self.linear2.weight.data = self.quantise(self.linear2.weight.data,self.bit_W,do_quantise=do_quantise)
        self.classifier.weight.data = self.quantise(self.classifier.weight.data,self.bit_W,do_quantise=do_quantise)
        
    def error_initialiser(self) :
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(1*5*5)]))
        self.error_conv1=n.sample((6,28,28,)).view(6,28,28).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(6*5*5)]))
        self.error_conv2=n.sample((16,10,10,)).view(16,10,10).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(120)]))
        self.error_linear1=n.sample((120,)).view(120).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(84)]))
        self.error_linear2=n.sample((84,)).view(84).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(10)]))
        self.error_classifier=n.sample((10,)).view(10).cuda()
        

 