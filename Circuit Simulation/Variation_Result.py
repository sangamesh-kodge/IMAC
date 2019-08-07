#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 22:13:20 2019

@author: skodge
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import pylab
import sys

import pandas as pd 
a =np.loadtxt("/home/nano01/a/agrawa64/cadence_folder/sangamesh/Sangamesh_SRAM/6T_MAC/spectre/mc_mult/psf/analog_multData.txt")
mult=np.zeros((16,16,1000))
for it in range (1000): 
    k=0
    for i in range(16):
        for j in range(16):
            mult[i][j][it] = a[it][k]
            k=k+1    
mult = np.array(mult)
W=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
mult =1.2-mult
mult_mean=np.mean(mult, axis=2)
mult_std=np.std(mult, axis=2)
mean = np.asarray(mult_mean)
np.savetxt("/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/Circuit Simulation/mean.csv", mean, delimiter=",")
std = np.asarray(mult_std)
np.savetxt("/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/Circuit Simulation/std.csv", std, delimiter=",")

matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
mu=mean[0][0]
sigma=std[0][0]
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
plt.plot(x, stats.norm.pdf(x, mu, sigma),label="W=0 Vin=0")

mu=mean[5][5]
sigma=std[5][5]
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
plt.plot(x, stats.norm.pdf(x, mu, sigma),label="W=5 Vin=5")

mu=mean[5][10]
sigma=std[5][10]
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000000)
plt.plot(x, stats.norm.pdf(x, mu, sigma),label="W=10 Vin=5")

mu=mean[10][10]
sigma=std[10][10]
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000000)
plt.plot(x, stats.norm.pdf(x, mu, sigma),label="W=10 Vin=10")
mu=mean[10][15]
sigma=std[10][15]
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000000)
plt.plot(x, stats.norm.pdf(x, mu, sigma),label="W=15 Vin=10") 

mu=mean[15][15]
sigma=std[15][15]
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000000)
plt.plot(x, stats.norm.pdf(x, mu, sigma),label="W=15 Vin=15")

plt.xlabel('analog product')
plt.ylabel('Probability density')
plt.title('Variation Results')
plt.legend()
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/Circuit Simulation/variation_analog_mult_pdf.png',bbox_inches='tight')
plt.show()


matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
k=0
s=np.zeros(256)
m=np.zeros(256)
for i in range(16):
    for j in range(16):
        m[k]=mean[i][j]
        s[k]=std[i][j]
        k=k+1
plt.scatter (m,s,c='red',marker='x')
plt.xlabel('mean')
plt.ylabel('Sigma')
plt.title('Variation Results')
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/Circuit Simulation/variation_analog_mult_mean_sigma.png',bbox_inches='tight')
plt.show()


matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
a = np.loadtxt("/home/nano01/a/agrawa64/cadence_folder/sangamesh/Sangamesh_SRAM/6T_MAC/spectre/mc_mac/psf_0/digital_macData.txt")
b = np.loadtxt("/home/nano01/a/agrawa64/cadence_folder/sangamesh/Sangamesh_SRAM/6T_MAC/spectre/mc_mac/psf_r2/digital_macData.txt")
c = np.loadtxt("/home/nano01/a/agrawa64/cadence_folder/sangamesh/Sangamesh_SRAM/6T_MAC/spectre/mc_mac/psf_r/digital_macData.txt")
d = np.loadtxt("/home/nano01/a/agrawa64/cadence_folder/sangamesh/Sangamesh_SRAM/6T_MAC/spectre/mc_mac/psf_15/digital_macData.txt")
plt.hist(a, bins=[-0.5,0.5,1.5,2.5,3.5],edgecolor='black', linewidth=1.2,color='skyblue', label ="Nominal=0")
plt.hist(b, bins=[3.5,4.5,5.5,6.5,7.5,8.5],edgecolor='black', linewidth=1.2,color='orange',label ="Nominal=6")
plt.hist(c, bins=[8.5,9.5,10.5,11.5,12.5,13.5],edgecolor='black', linewidth=1.2,color='yellow',label ="Nominal=11")
plt.hist(d, bins=[12.5,13.5,14.5,15.5],edgecolor='black', linewidth=1.2,color='lightgreen',label ="Nominal=15")
plt.xlabel("MAC Output")
plt.ylabel("# of Occurrences")
plt.title("1000 runs of MC",y=1.18)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
          ncol=4, fancybox=True, shadow=True)
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/Circuit Simulation/MC.png',bbox_inches='tight')
plt.show()


