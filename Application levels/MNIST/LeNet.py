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
    def __init__(self,input_size, minacc=0.12,maxacc=0.45,bit_W=4, bit_A=4, sigma=0.0, n_classes=10, **kwargs):
        self.bit_A=bit_A
        self.bit_W=bit_W
        self.sigma=sigma
        self.minacc = minacc
        self.maxacc = maxacc 
        self.input_size=input_size
        self.n_classes=n_classes
        
        
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
    
    def quantise(self, x, k, do_quantise=False):
        Max=torch.max(x)
        if( not do_quantise):
            Digital=x
            return Digital
        Digital=torch.round(((2**k)-1)*x/Max)
        
        #output=Min+(Max-Min)*Digital/((2**k)-1)
        return Digital
    
    def inference(self, x,do_quantise = True):
        #layer 1
        x = self.myconv(inp=x, w=self.conv1.weight.data, input_channels=self.input_size, output_channels=6, kernel_size=5, padding=2)
        
        x = F.relu(x)
        x = self.maxpool1(x)
        #layer 2
        x,d = self.quantise(x,self.bit_A, do_quantise)
        x = self.myconv(inp=x,digital=d,weight=self.conv2.weight.data,input_channels=6,output_channels=16, kernel_size=5, padding=0, bias=self.conv2.bias.data)
        
        x = F.relu(x)
        x = self.maxpool2(x)
        """
        #Flatten
        x = x.view(x.size(0), -1)
        #layer 3
        x,d = self.quantise(x,self.bit_A, do_quantise)
        x = self.mylinear(inp=x,digital=d,weight=self.linear1.weight.data,input_channels=400,output_channels=120, bias=self.linear1.bias.data)
        x = F.relu(x)
        #layer 4
        x,d = self.quantise(x,self.bit_A, do_quantise)
        x = self.mylinear(inp=x,digital=d,weight=self.linear2.weight.data,input_channels=84,output_channels=self.out, bias=self.linear2.bias.data)
        x = F.relu(x)
        #layer 5
        x,d = self.quantise(x,self.bit_A, do_quantise)
        x = self.mylinear(inp=x,digital=d,weight=self.classifier.weight.data,input_channels=120,output_channels=self.n_classes, bias=self.classifier.bias.data)
        """
        return (x)
             
    
    
    def myconv(self,x,w,input_channels,output_channels,kernel_size,padding):
        #non_linearity matrix read from the file
        mult = pd.read_csv('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/Circuit Simulation/nominal.csv',header=None)
        mult = torch.tensor(mult.values)
        #storing min and max result for scaling at the end 
        macmin = torch.max(self.conv1(x))
        macmax = torch.min(self.conv1(x))
        
        # padding
        p = torch.nn.ConstantPad2d((padding,padding,padding,padding),0)
        x = p(x)
        
        # define output map
        s = [x.size()[0],x.size()[1],x.size()[2],x.size()[3]]
        Nmovx=s[3]-kernel_size+1
        Nmovy=s[2]-kernel_size+1
        activation_vector=torch.zeros(s[0],output_channels,Nmovy,Nmovx)
        
        # flatten weight
        w = w.view(w.size()[0],-1)
         
        # separating positive form negative 
        xp = torch.zeros(x.size()).cuda()
        xn = torch.zeros(x.size()).cuda()
        xp = torch.where(x>0,x,xp)
        xn = torch.where(x<0,-x,xn)
        xdp =self.quantise(xp,self.bit_A,do_quantise=True)
        xdn =self.quantise(xn,self.bit_A,do_quantise=True)
        wp = torch.zeros(w.size()).cuda()
        wn = torch.zeros(w.size()).cuda()
        wp = torch.where(w>0,w,wp)
        wn = torch.where(w<0,-w,wn)
        wdp =self.quantise(wp,self.bit_W,do_quantise=True)
        wdn =self.quantise(wn,self.bit_W,do_quantise=True)
        
        for i in range (Nmovy):
            for j in range(Nmovx):
                # slice actiation and flatten
                adp = xdp[:,:,i:i+kernel_size,j:j+kernel_size].contiguous().view(x.size(0),-1)
                adn = xdn[:,:,i:i+kernel_size,j:j+kernel_size].contiguous().view(x.size(0),-1)
                k = 0
                maxk = adp.size()[1]
                while ((maxk+1-k !=0 and (maxk+1)%25==0) or((maxk-k>25 and (maxk+1)%25!=0))):
                    for batch in range(s[0]):
                        for o_c in range(output_channels):
                            #accumulation without proper scaling
                            virtual_acc25p=torch.sum(mult[wdp[o_c,k:k+25],adp[batch,k:k+25]]).view(-1) + torch.sum(mult[wdn[o_c,k:k+25],adn[batch,k:k+25]]).view(-1)      
                            virtual_acc25n=torch.sum(mult[wdn[o_c,k:k+25],adp[batch,k:k+25]]).view(-1) + torch.sum(mult[wdp[o_c,k:k+25],adn[batch,k:k+25]]).view(-1)      
                            # scaling the result of accumulation
                            acc25p = (self.maxacc-self.minacc)*(virtual_acc25p-25*mult.min())/(25*mult.max()-25*mult.min())+self.minacc
                            acc25n = (self.maxacc-self.minacc)*(virtual_acc25n-25*mult.min())/(25*mult.max()-25*mult.min())+self.minacc
                            # digital output
                            digital_outp = torch.round((acc25p-self.minacc)*(2**4-1) /(self.maxacc-self.minacc))
                            digital_outn = torch.round((acc25n-self.minacc)*(2**4-1) /(self.maxacc-self.minacc))
                            # accumulating the 25 vector mac
                            activation_vector[batch,o_c,i,j] = activation_vector[batch,o_c,i,j] +(digital_outp-digital_outn)
                            k=k+25
                for batch in range(s[0]):
                    for o_c in range(output_channels):
                        #accumulation without proper scaling
                        virtual_acc25p=torch.sum(mult[wdp[o_c,k:maxk],adp[batch,k:maxk]]).view(-1) + torch.sum(mult[wdn[o_c,k:maxk],adn[batch,k:maxk]]).view(-1)      
                        virtual_acc25n=torch.sum(mult[wdn[o_c,k:maxk],adp[batch,k:maxk]]).view(-1) + torch.sum(mult[wdp[o_c,k:maxk],adn[batch,k:maxk]]).view(-1)      
                        # scaling the result of accumulation
                        acc25p = (self.maxacc-self.minacc)*(virtual_acc25p-(maxk-k)*mult.min())/(25*mult.max()-25*mult.min())+self.minacc
                        acc25n = (self.maxacc-self.minacc)*(virtual_acc25n-(maxk-k)*mult.min())/(25*mult.max()-25*mult.min())+self.minacc
                        # digital output
                        digital_outp = torch.round((acc25p-self.minacc)*(2**4-1) /(self.maxacc-self.minacc))
                        digital_outn = torch.round((acc25n-self.minacc)*(2**4-1) /(self.maxacc-self.minacc))
                        # accumulating the 25 vector mac
                        activation_vector[batch,o_c,i,j] = activation_vector[batch,o_c,i,j] +(digital_outp-digital_outn)
        activation_vector = (activation_vector-activation_vector.min())*(macmax-macmin)/((activation_vector.max()-activation_vector.min()))+macmin
        return activation_vector
    
                                    
                

                
                
        
        
        
        
        
        
        
    
                