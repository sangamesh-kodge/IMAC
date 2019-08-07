#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 03:02:30 2019

@author: skodge
"""

import numpy as np
import matplotlib.pyplot as plt

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
        
a=np.array([[2763,		125436,	3936.84375,	29409.44843,	689.0625,	40890.696],
[6685.714286,	266880,	8034.375,	60019.2825,	1406.25,	83450.4],
[25097.14286,	484800,	2250,		219845.4,	281.25,		16690.08],
[5270.4, 	    101808,	472.5,		46167.534,	59.1796875,	3504.9168],
[439.2,		    8484,	39.375,		3847.2945,	4.98046875,	292.0764],
[40255.45714,	987408,	14733.09375,	359288.9594,	2440.722656,	144828.1692]])

a=a.T/1000




# data to plot
n_groups = 6

tvn=a[0]/a[0]
tdima=a[2]/a[0]
tinmemory=a[4]/a[0]

# create plot
matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))

index = np.arange(n_groups)
bar_width = 0.3
opacity = 1.0

rects1 = plt.bar(index, tvn, bar_width,
alpha=opacity,
color='b',
label='Von Neumann')

rects2 = plt.bar(index + bar_width, tdima, bar_width,
alpha=opacity,
color='g',
label='DIMA')

rects3 = plt.bar(index + 2*bar_width, tinmemory, bar_width,
alpha=opacity,
color='brown',
label='This Work')

plt.xlabel('Layers')
plt.ylabel('Normalised Delay')
plt.title('Normalised Delay Comparisons',y=1.18)
plt.xticks(index + bar_width, ('C1', 'C2', 'FC1', 'FC2','FC3','Total'))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
          ncol=3, fancybox=True, shadow=True)
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/System/Delay_norm.png',bbox_inches='tight')
plt.show()
autolabel(rects1)


tvn=a[0]
tdima=a[2]
tinmemory=a[4]

# create plot
matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
index = np.arange(n_groups)
bar_width = 0.3
opacity = 1.0
rects1 = plt.bar(index, tvn, bar_width,
alpha=opacity,
color='b',
label='Von Neumann')
rects2 = plt.bar(index + bar_width, tdima, bar_width,
alpha=opacity,
color='g',
label='DIMA')
rects3 = plt.bar(index + 2*bar_width, tinmemory, bar_width,
alpha=opacity,
color='brown',
label='This Work')
plt.xlabel('Layers')
plt.ylabel('Delay( us ) / decision')
plt.title('Delay Comparisons',y=1.18)
plt.xticks(index + bar_width, ('C1', 'C2', 'FC1', 'FC2','FC3','Total'))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
          ncol=3, fancybox=True, shadow=True)
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/System/Delay.png',bbox_inches='tight')
plt.show()





evn=a[1]/a[1]
edima=a[3]/a[1]
einmemory=a[5]/a[1]
matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
index = np.arange(n_groups)
bar_width = 0.3
opacity = 1.0

rects1 = plt.bar(index, evn, bar_width,
alpha=opacity,
color='b',
label='Von Neumann')

rects2 = plt.bar(index + bar_width, edima, bar_width,
alpha=opacity,
color='g',
label='DIMA')

rects3 = plt.bar(index + 2*bar_width, einmemory, bar_width,
alpha=opacity,
color='brown',
label='This Work')

plt.xlabel('Layers')
plt.ylabel('Normalised Energy ')
plt.title('Normalised Energy Comparisons',y=1.18)
plt.xticks(index + bar_width, ('C1', 'C2', 'FC1', 'FC2','FC3','Total'))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
          ncol=3, fancybox=True, shadow=True)
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/System/Energy_norm.png',bbox_inches='tight')
plt.show()

evn=a[1]
edima=a[3]
einmemory=a[5]
matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
index = np.arange(n_groups)
bar_width = 0.3
opacity = 1.0

rects1 = plt.bar(index, evn, bar_width,
alpha=opacity,
color='b',
label='Von Neumann')

rects2 = plt.bar(index + bar_width, edima, bar_width,
alpha=opacity,
color='g',
label='DIMA')

rects3 = plt.bar(index + 2*bar_width, einmemory, bar_width,
alpha=opacity,
color='brown',
label='This Work')

plt.xlabel('Layers')
plt.ylabel('Energy( nJ )/decision')
plt.title('Energy Comparisons',y=1.18)
plt.xticks(index + bar_width, ('C1', 'C2', 'FC1', 'FC2','FC3','Total'))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
          ncol=3, fancybox=True, shadow=True)
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/System/Energy.png',bbox_inches='tight')
plt.show()


edvn=a[1]*a[0]/(a[0]*a[1])
eddima=a[3]*a[2]/(a[0]*a[1])
edinmemory=a[5]*a[4]/(a[0]*a[1])
matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
index = np.arange(n_groups)
bar_width = 0.3
opacity = 1.0

rects1 = plt.bar(index, edvn, bar_width,
alpha=opacity,
color='b',
label='Von Neumann')

rects2 = plt.bar(index + bar_width, eddima, bar_width,
alpha=opacity,
color='g',
label='DIMA')

rects3 = plt.bar(index + 2*bar_width, edinmemory, bar_width,
alpha=opacity,
color='brown',
label='This Work')

plt.xlabel('Layers')
plt.ylabel('Normalised EDP/decision')
plt.title('Normalised EDP Comparisons',y=1.18)
plt.xticks(index + bar_width, ('C1', 'C2', 'FC1', 'FC2','FC3','Total'))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
          ncol=3, fancybox=True, shadow=True)
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/System/Energy_delay_norm.png',bbox_inches='tight')
plt.show()


edvn=a[1]*a[0]/1000
eddima=a[3]*a[2]/1000
edinmemory=a[5]*a[4]/1000
matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
index = np.arange(n_groups)
bar_width = 0.3
opacity = 1.0

rects1 = plt.bar(index, edvn, bar_width,
alpha=opacity,
color='b',
label='Von Neumann')

rects2 = plt.bar(index + bar_width, eddima, bar_width,
alpha=opacity,
color='g',
label='DIMA')

rects3 = plt.bar(index + 2*bar_width, edinmemory, bar_width,
alpha=opacity,
color='brown',
label='This Work')

plt.xlabel('Layers')
plt.ylabel('EDP( p Js )/decision')
plt.title('EDP Comparisons',y=1.18)
plt.xticks(index + bar_width, ('C1', 'C2', 'FC1', 'FC2','FC3','Total'))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
          ncol=3, fancybox=True, shadow=True)
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/System/Energy_delay.png',bbox_inches='tight')
plt.show()