#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 03:02:30 2019

@author: skodge
"""

import numpy as np

a=np.array([[4043.71875	,153660,	3936.84375,	29400,	689.0625	, 43417.92],
[8416.071429, 	324480,	8034.375,	60000	,1406.25	,88608],
[5395.714286,	312000	,1606.875,	196320,	281.25,	17721.6],
[1133.1,	65520,	337.44375,	41227.2,	59.1796875,	3721.536],
[94.425,	5460,	28.1203125,	3435.6,	4.98046875,	310.128]])
a=a.T



import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 5

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
plt.title('Delay Comparisons Normalised')
plt.xticks(index + bar_width, ('C1', 'C2', 'FC1', 'FC2','FC3'))
plt.legend()
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/System/Delay_norm.png')
plt.show()



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
plt.ylabel('Delay per decision')
plt.title('Delay Comparisons')
plt.xticks(index + bar_width, ('C1', 'C2', 'FC1', 'FC2','FC3'))
plt.legend()
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/System/Delay.png')
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
plt.title('Energy Comparisons Normalised')
plt.xticks(index + bar_width, ('C1', 'C2', 'FC1', 'FC2','FC3'))
plt.legend()
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/System/Energy_norm.png')
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
plt.ylabel('Energy per decision')
plt.title('Energy Comparisons')
plt.xticks(index + bar_width, ('C1', 'C2', 'FC1', 'FC2','FC3'))
plt.legend()
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/System/Energy.png')
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
plt.ylabel('Normalised Energy*delay per decision')
plt.title('Energy*delay Comparisons Normalised')
plt.xticks(index + bar_width, ('C1', 'C2', 'FC1', 'FC2','FC3'))
plt.legend()
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/System/Energy_delay_norm.png')
plt.show()


edvn=a[1]*a[0]
eddima=a[3]*a[2]
edinmemory=a[5]*a[4]
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
plt.ylabel('Energy*delay per decision')
plt.title('Energy*delay Comparisons')
plt.xticks(index + bar_width, ('C1', 'C2', 'FC1', 'FC2','FC3'))
plt.legend()
plt.savefig('/home/min/a/skodge/Project/GitHub/6T-SRAM-Multiplication/System/Energy_delay.png')
plt.show()