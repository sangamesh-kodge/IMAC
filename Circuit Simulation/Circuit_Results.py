#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 22:13:20 2019

@author: skodge
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats


a =np.loadtxt('/home/nano01/a/agrawa64/cadence_folder/sangamesh/Sangamesh_SRAM/6T_MAC/spectre/trans_mult/nominal.txt')
mult=np.zeros((16,16))

k=0
for i in range(16):
    for j in range(16):
        mult[i][j] = a[k]
        k=k+1
mult= np.array(mult)
mult=1.2-mult
np.savetxt("/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/Circuit Simulation/nominal.csv", mult, delimiter=",")

        
W=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])


multW=np.array(mult[[0,5,10,15]][:])
multV=np.array(mult.T[[0,5,10,15]][:])

matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
y_bl=multW
plt.ylabel('Analog Output')
plt.xlabel('Vin')
slope, intercept, r_value, p_value, std_err = stats.linregress(W,y_bl[0,:])
line = slope*W+intercept
plt.plot(W,line,c='red')
plt.scatter(W,y_bl[0,:].T,marker="o",c='red')
slope, intercept, r_value, p_value, std_err = stats.linregress(W,y_bl[1,:].reshape(16,))
line = slope*W+intercept
plt.plot(W,line,c='green')
plt.scatter(W,y_bl[1,:].T,marker="o",c='green')
slope, intercept, r_value, p_value, std_err = stats.linregress(W,y_bl[2,:].reshape(16,))
line = slope*W+intercept
plt.plot(W,line,c='blue')
plt.scatter(W,y_bl[2,:].T,marker="o",c='blue')
slope, intercept, r_value, p_value, std_err = stats.linregress(W,y_bl[3,:].reshape(16,))
line = slope*W+intercept
plt.plot(W,line,c='orange')
plt.scatter(W,y_bl[3,:].T,marker="o",c='orange')
plt.legend(('W=0','W=5','W=10','W=15'))
plt.title("Output with Vin sweep")
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/Circuit Simulation/OutVsVin.png',bbox_inches='tight')

matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
y_bl=multV
plt.ylabel('Analog Output')
plt.xlabel('W')
slope, intercept, r_value, p_value, std_err = stats.linregress(W,y_bl[0,:])
line = slope*W+intercept
plt.plot(W,line,c='red')
plt.scatter(W,y_bl[0,:].T,marker="o",c='red')
slope, intercept, r_value, p_value, std_err = stats.linregress(W,y_bl[1,:].reshape(16,))
line = slope*W+intercept
plt.plot(W,line,c='green')
plt.scatter(W,y_bl[1,:].T,marker="o",c='green')
slope, intercept, r_value, p_value, std_err = stats.linregress(W,y_bl[2,:].reshape(16,))
line = slope*W+intercept
plt.plot(W,line,c='blue')
plt.scatter(W,y_bl[2,:].T,marker="o",c='blue')
slope, intercept, r_value, p_value, std_err = stats.linregress(W,y_bl[3,:].reshape(16,))
line = slope*W+intercept
plt.plot(W,line,c='orange')
plt.scatter(W,y_bl[3,:].T,marker="o",c='orange')
plt.legend(('Vin=0','Vin=5','Vin=10','Vin=15'))
plt.title("Output with W sweep")
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/Circuit Simulation/OutVsW.png',bbox_inches='tight')

matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))

x1 = np.zeros((1*15))
y1 = np.zeros((1*15))
k=0
for i in range(0,2,1):
    for j in range(16):
        if(i*j!=0  ):
            x1[k] = i*j 
            y1[k]=mult[i][j]
            k=k+1

x2 = np.zeros((2*15))
y2 = np.zeros((2*15))
k=0
for i in range(2,4,1):
    for j in range(16):
        if(i*j!=0  ):
            x2[k] = i*j 
            y2[k]=mult[i][j]
            k=k+1
        
x3 = np.zeros((4*15))
y3 = np.zeros((4*15))
k=0
for i in range(4,8,1):
    for j in range(16):
        if(i*j!=0  ):
            x3[k] = i*j 
            y3[k]=mult[i][j]
            k=k+1

x4 = np.zeros((8*15))
y4 = np.zeros((8*15))
k=0
for i in range(8,16,1):
    for j in range(16):
        if(i*j!=0  ):
            x4[k] = i*j 
            y4[k]=mult[i][j]
            k=k+1
            
x = np.zeros((16*16))
y = np.zeros((16*16))
k=0
for i in range(16):
    for j in range(16):
        x[k] = i*j 
        y[k]=mult[i][j]
        k=k+1
        
                        
plt.ylabel('Observed analog output')
plt.xlabel('Digital output')
plt.title("Non-linearity")

slope, intercept, r_value, p_value, std_err = stats.linregress(x1,y1)
line1 = slope*x1+intercept
slope, intercept, r_value, p_value, std_err = stats.linregress(x2,y2)
line2 = slope*x2+intercept
slope, intercept, r_value, p_value, std_err = stats.linregress(x3,y3)
line3 = slope*x3+intercept
slope, intercept, r_value, p_value, std_err = stats.linregress(x4,y4)
line4 = slope*x4+intercept

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
line = slope*x+intercept

#plt.plot([0,1*15],[y.min(),y.max()],label = "Expected")
'''
plt.plot(x1,line1,color="blue", label = "line (W=0-1)")
plt.plot(x2,line2,color="green", label = "line (W=2-3)")
plt.plot(x3,line3,color="red", label = "line (W=4-7)")
plt.plot(x4,line4,color="orange", label = "line (W=8-16)")
'''
plt.plot(x,line,color="blue", label = "linear fit")
plt.scatter(x,y, marker="x",color="red",label="Observed")
plt.legend()
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/Circuit Simulation/Non-linearity.png',bbox_inches='tight')


