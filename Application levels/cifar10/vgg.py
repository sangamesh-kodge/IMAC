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
    def __init__(self,input_size,bit_W=4, bit_A=4, sigma=0.0, n_classes=10, **kwargs):
        self.bit_A=bit_A
        self.bit_W=bit_W
        self.sigma=sigma
        self.bit_out=4
        self.noofacc=10
        
        self.input_size=input_size
        self.n_classes=n_classes
        self.quantise_weight_flag= False
        self.error_initialiser() 
        self.input_size=input_size    
        
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


    def forward(self, x, training= True):
        x = F.relu(self.conv1(x))
        if (training):
            x = F.dropout(x,.3)  
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)

        x = F.relu(self.conv3(x))
        if (training):
            x = F.dropout(x,.4)  
        x = F.relu(self.conv4(x))
        x = self.maxpool2(x)        

        x = F.relu(self.conv5(x))
        if (training):
            x = F.dropout(x,.4)  
        x = F.relu(self.conv6(x))
        if (training):
            x = F.dropout(x,.4)  
        x = F.relu(self.conv7(x))
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        if (training):
            x = F.dropout(x,.5)    
        x = F.relu(self.linear2(x))
        if (training):
            x = F.dropout(x,.5)    
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

    def inference(self, x, do_quantise = True, do_add_var =True):
        if (do_quantise and (not self.quantise_weight_flag)):
            self.quantise_weight_flag=True
            self.quantise_weight()
        #layer 1     
        x = F.relu(self.add_variations( self.conv1(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_conv1 , self.bit_out,input_channels=self.input_size,kernel_size=3, do_add_var=do_add_var ))
        #layer 2
        x = F.relu(self.add_variations( self.conv2(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_conv2 , self.bit_out,input_channels=64,kernel_size=3, do_add_var=do_add_var ))
        #maxpool
        x = self.maxpool1(x)
 
        #layer 3
        x = F.relu(self.add_variations( self.conv3(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_conv3 , self.bit_out,input_channels=64,kernel_size=3, do_add_var=do_add_var ))
        #layer 4
        x = F.relu(self.add_variations( self.conv4(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_conv4 , self.bit_out,input_channels=128,kernel_size=3, do_add_var=do_add_var ))
        x = self.maxpool2(x)  
        #layer 5
        x = F.relu(self.add_variations( self.conv5(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_conv5 , self.bit_out,input_channels=128,kernel_size=3, do_add_var=do_add_var ))
        #layer 6
        x = F.relu(self.add_variations( self.conv6(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_conv6 , self.bit_out,input_channels=256,kernel_size=3, do_add_var=do_add_var ))
        #layer 7
        x = F.relu(self.add_variations( self.conv7(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_conv7 , self.bit_out,input_channels=256,kernel_size=3, do_add_var=do_add_var ))
        x = self.maxpool3(x)

        x = x.view(x.size(0), -1)
        #linear1
        x = F.relu(self.add_variations( self.linear1(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_linear1 ,self.bit_out,input_channels=4096,kernel_size=1,do_add_var=do_add_var ))
        #linear2
        x = F.relu(self.add_variations( self.linear2(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_linear2 , self.bit_out,input_channels=4096,kernel_size=1, do_add_var=do_add_var ))
        #classifier
        x = self.add_variations( self.classifier(self.quantise(x,self.bit_A,do_quantise=do_quantise)), self.error_classifier , self.bit_out, input_channels=4096, kernel_size=1, do_add_var=do_add_var )
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
        
    
    def quantise_weight(self,do_quantise=True):
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
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(3*3*3/self.noofacc)]))
        self.error_conv1=n.sample((64,32,32,)).view(64,32,32).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(64*3*3/self.noofacc)]))
        self.error_conv2=n.sample((64,32,32,)).view(64,32,32).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(64*3*3/self.noofacc)]))
        self.error_conv3=n.sample((128,16,16,)).view(128,16,16).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(128*3*3/self.noofacc)]))
        self.error_conv4=n.sample((128,16,16,)).view(128,16,16).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(128*3*3/self.noofacc)]))
        self.error_conv5=n.sample((256,8,8,)).view(256,8,8).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(256*3*3/self.noofacc)]))
        self.error_conv6=n.sample((256,8,8,)).view(256,8,8).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(256*3*3/self.noofacc)]))
        self.error_conv7=n.sample((256,8,8,)).view(256,8,8).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(4096/self.noofacc)]))
        self.error_linear1=n.sample((4096,)).view(4096).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(4096/self.noofacc)]))
        self.error_linear2=n.sample((4096,)).view(4096).cuda()
        n=torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([self.sigma*math.sqrt(10/self.noofacc)]))
        self.error_classifier=n.sample((10,)).view(10).cuda()
        

 
