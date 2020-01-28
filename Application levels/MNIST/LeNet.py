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
import numpy as np
import pandas as pd

#import sys
        
    

class lenet(nn.Module):
    def __init__(self,input_size, minacc=0.17,maxacc=0.45,bit_W=4, bit_A=4, sigma=0.0, n_classes=10, **kwargs):
        self.bit_A=bit_A
        self.bit_W=bit_W
        self.sigma=sigma
        self.bit_out=4
        self.noofacc=25
        
        self.minacc = minacc
        self.maxacc = maxacc 
        
        self.input_size=input_size
        self.n_classes=n_classes
        self.quantise_weight_flag= False
        self.error_initialiser()        
        
        
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 6, kernel_size=5, padding=2, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 =nn.Linear(400,120,bias=True)
        self.linear2 =nn.Linear(120,84,bias=True)
        self.classifier = nn.Linear(84, n_classes, bias=True)
      
        
        
        
    def forward(self, x, training = True ):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        if (training):
            x = F.dropout(x,.2)  
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)        
        if (training):
            x = F.dropout(x,.2)  
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        if (training):
            x = F.dropout(x,.2)    
        x = F.relu(self.linear2(x))
        if (training):
            x = F.dropout(x,.2)    
        x = self.classifier(x)
        return x
    
    def quantise(self, x, k, do_quantise=False):
        Max=torch.max(x)
        Min=torch.min(x)
        if Max<-Min:
            Max=-Min
        if( not do_quantise):
            return x
        Digital=torch.round(((2**k)-1)*x/Max)
        output=Max*Digital/((2**k)-1)
        return output
    
    def quantise_weight(self, do_quantise=True):
        self.conv1.weight.data = self.quantise(self.conv1.weight.data,self.bit_W,do_quantise=do_quantise)
        self.conv2.weight.data = self.quantise(self.conv2.weight.data,self.bit_W,do_quantise=do_quantise)
        self.linear1.weight.data = self.quantise(self.linear1.weight.data,self.bit_W,do_quantise=do_quantise)
        self.linear2.weight.data = self.quantise(self.linear2.weight.data,self.bit_W,do_quantise=do_quantise)
        self.classifier.weight.data = self.quantise(self.classifier.weight.data,self.bit_W,do_quantise=do_quantise)
    
    def inference(self, x, do_quantise = True, do_add_var =True):
        if (do_quantise and (not self.quantise_weight_flag)):
            self.quantise_weight_flag=True
            self.quantise_weight()
        #layer 1     
        x = F.relu(self.add_variations( self.conv1(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_conv1 , self.bit_out,input_channels=1,kernel_size=5, do_add_var=do_add_var ))
        x = self.maxpool1(x)
        #layer 2
        x = F.relu(self.add_variations( self.conv2(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_conv2 , self.bit_out,input_channels=6,kernel_size=5, do_add_var=do_add_var ))
        x = self.maxpool2(x)        
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.add_variations( self.linear1(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_linear1 ,self.bit_out,input_channels=400,kernel_size=1,do_add_var=do_add_var ))
        x = F.relu(self.add_variations( self.linear2(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_linear2 , self.bit_out,input_channels=120,kernel_size=1, do_add_var=do_add_var ))
        x = self.add_variations( self.classifier(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_classifier , self.bit_out,input_channels=84,kernel_size=1,do_add_var=do_add_var )
        return (x)
    
    def add_variations(self,x,error,k, input_channels, kernel_size,  do_add_var=True):
        if(not do_add_var):
            return x
        Max=torch.max(x)
        Min=torch.min(x)
        no_of_terms = input_channels*kernel_size*kernel_size/self.noofacc
        levels= (2**k-1)*no_of_terms
            
        if (Min<0):
            outp = torch.zeros(x.size()).cuda()
            outn = torch.zeros(x.size()).cuda()
            error_out = torch.round(error)
            outp = torch.round(torch.where(x>0,x,outp)*levels/Max)
            outn = torch.round(torch.where(x<0,-x,outn)*levels/Min)
            
            out= (outp+outn)+error_out
            
            outp=torch.zeros(x.size()).cuda()
            outn=torch.zeros(x.size()).cuda()
            
            outp=torch.clamp(torch.where(x>0,out,outp),0,levels)*Max/levels
            outn=torch.clamp(torch.where(x<0,out,outn),0,levels)*Min/levels
            
            out1=outp-outn
        else:
            outp=x
            error_out = torch.round(error)
            outp = torch.round(torch.where(x>0,x,outp)*levels/Max)
            out= outp+error_out
            outp=torch.zeros(x.size()).cuda()
            outp=torch.clamp(torch.where(x>0,out,outp),0,levels)*Max/levels
            out1=outp
        return out1 
    
    def error_initialiser(self) :
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(1*5*5/self.noofacc)]))
        self.error_conv1=n.sample((6,28,28,)).view(6,28,28).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(6*5*5/self.noofacc)]))
        self.error_conv2=n.sample((16,10,10,)).view(16,10,10).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(400/self.noofacc)]))
        self.error_linear1=n.sample((120,)).view(120).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(120/self.noofacc)]))
        self.error_linear2=n.sample((84,)).view(84).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(84/self.noofacc)]))
        self.error_classifier=n.sample((10,)).view(10).cuda()
                
                
        
        
        
        
        
        
        
    
                
