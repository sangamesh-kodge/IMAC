#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:35:30 2019

@author: skodge
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
'''


y_bl=np.array([[31,	29,	27,	25,	23,	21,	19,	16,	19,	16,	15,	12,	11,	8,	5,	1],
[29,	27,	26,	23,	23,	20,	18,	15,	17,	15,	14,	11,	10,	7,	5,	1],
[28,	26,	25,	22,	21,	19,	17,	15,	16,	15,	13,	11,	9,	7,	4,	1],
[27,	25,	23,	21,	20,	17,	16,	15,	15,	14,	12,	10,	9,	6,	4,	1],
[25,	23,	21,	19,	18,	16,	15,	14,	15,	13,	11,	9,	8,	5,	3,	1],
[23,	21,	19,	17,	16,	15,	15,	13,	13,	11,	10,	8,	7,	5,	3,	1],
[20,	18,	17,	15,	15,	15,	13,	11,	12,	10,	9,	7,	6,	4,	3,	1],
[17,	16,	15,	15,	14, 13, 12, 10,	11,	9,	8,	6,	5,	4,	3,	1],
[15,	15,	14,	13,	12,	11,	10,	9,	9,	8,	7,	5,	5,	3,	2,	1],
[14,	13,	12,	11,	10,	9,	9,	7,	8,	7,	6,	5,	4,	3,	2,	1],
[12,	11,	10,	9,	9,	8,	7,	6,	6,	5,	5,	4,	3,	2,	1,	1],
[9,	9,	8,	7,	7,	6,	5,	5,	5,	4,	4,	3,	3,	2,	1,	1],
[7,	7,	6,	5,	5,	4,	4,	3,	3,	3,	3,	2,	2,	1,	1,	0],
[5,	5,	4,	3,	3,	3,	3,	3,	3,	2,	2,	1,	1,	1,	1,	0],
[3,	3,	3,	2,	2,	2,	2,	2,	2,	1,	1,	1,	1,	1,	0,	0],
[2,	2,	2,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0]])
Vin=np.array([15, 14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
W=np.array([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
plt.plot(Vin,y_bl)
plt.ylabel('Digital Output')
plt.xlabel('Vin')
plt.legend(('W=15','W=14','W=13','W=12','W=11','W=10','W=9','W=8','W=7','W=6','W=5','W=4','W=3','W=2','W=1','W=0'))


plt.figure()
plt.plot(W,y_bl.T)
plt.ylabel('Digital Output')
plt.xlabel('2nd Operand')
plt.legend(('Vin=15','Vin=14','Vin=13','Vin=12','Vin=11','Vin=10','Vin=9','Vin=8','Vin=7','Vin=6','Vin=5','Vin=4','Vin=3','Vin=2','Vin=1','Vin=0'))


product= (Vin.reshape(16,1)*W.reshape(1,16))*31/225
product= np.round(product.reshape(256,))
plt.figure()
plt.plot(product,product)
plt.scatter(product,y_bl.reshape(256,))
'''
matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
y_bl=np.array([[31,	29,	27,	25,	23,	21,	19,	16,	19,	16,	15,	12,	11,	8,	5,	1],
[23,	21,	19,	17,	16,	15,	15,	13,	13,	11,	10,	8,	7,	5,	3,	1],
[12,	11,	10,	9,	9,	8,	7,	6,	6,	5,	5,	4,	3,	2,	1,	1],
[2,	2,	2,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0]])
W=np.array([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
plt.ylabel('Digital Output')
plt.xlabel('W')
plt.plot(W,y_bl.T,'-o')
plt.legend(('Vin=15','Vin=10','Vin=5','Vin=0'))
plt.savefig('OutVsW.png')

plt.figure(figsize=(10,5))
y_bl=np.array([[31,	21,	15,	1],
[29,	20,	14,	1],
[28,	19,	13,	1],
[27,	17,	12,	1],
[25,	16,	11,	1],
[23,	15,	10,	1],
[20,	15,	9, 1],
[17,	13,	8,	1],
[15,	11,	7,	1],
[14,	9,	6,	1],
[12,	8,	5,	1],
[9,	6,	4,	1],
[7,	4,	3,	0],
[5,	3,	2,	0],
[3,	2,	1,	0],
[2,	1,	1,	0]])
Vin=np.array([15, 14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
plt.ylabel('Digital Output')
plt.xlabel('Vin')
plt.plot(Vin,y_bl,'-o')
plt.legend(('W=15','W=10','W=5','W=0'))
plt.savefig('OutVsVin.png')

plt.figure(figsize=(10,5))
y_bl=np.array([[31,	29,	27,	25,	23,	21,	19,	16,	19,	16,	15,	12,	11,	8,	5,	1],
[29,	27,	26,	23,	23,	20,	18,	15,	17,	15,	14,	11,	10,	7,	5,	1],
[28,	26,	25,	22,	21,	19,	17,	15,	16,	15,	13,	11,	9,	7,	4,	1],
[27,	25,	23,	21,	20,	17,	16,	15,	15,	14,	12,	10,	9,	6,	4,	1],
[25,	23,	21,	19,	18,	16,	15,	14,	15,	13,	11,	9,	8,	5,	3,	1],
[23,	21,	19,	17,	16,	15,	15,	13,	13,	11,	10,	8,	7,	5,	3,	1],
[20,	18,	17,	15,	15,	15,	13,	11,	12,	10,	9,	7,	6,	4,	3,	1],
[17,	16,	15,	15,	14, 13, 12, 10,	11,	9,	8,	6,	5,	4,	3,	1],
[15,	15,	14,	13,	12,	11,	10,	9,	9,	8,	7,	5,	5,	3,	2,	1],
[14,	13,	12,	11,	10,	9,	9,	7,	8,	7,	6,	5,	4,	3,	2,	1],
[12,	11,	10,	9,	9,	8,	7,	6,	6,	5,	5,	4,	3,	2,	1,	1],
[9,	9,	8,	7,	7,	6,	5,	5,	5,	4,	4,	3,	3,	2,	1,	1],
[7,	7,	6,	5,	5,	4,	4,	3,	3,	3,	3,	2,	2,	1,	1,	0],
[5,	5,	4,	3,	3,	3,	3,	3,	3,	2,	2,	1,	1,	1,	1,	0],
[3,	3,	3,	2,	2,	2,	2,	2,	2,	1,	1,	1,	1,	1,	0,	0],
[2,	2,	2,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0]])
Vin=np.array([15, 14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
W=np.array([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
product= (Vin.reshape(16,1)*W.reshape(1,16))*31/225
product= np.round(product.reshape(256,))
plt.ylabel('Digital Output')
plt.xlabel('Vin*W')
plt.plot(product,product)
plt.scatter(product,y_bl.reshape(256,),marker="x",c='red')
plt.legend(('Expected','Observed'))
plt.savefig('ExpVsObs.png')

plt.figure(figsize=(10,6))
matplotlib.rcParams.update({'font.size': 20})
plt.ylabel('Number of Samples')
plt.xlabel('Digital Output')
plt.hist([0,1,2,3,4,5],bins=[-.5,.5,1.5,2.5,3.5,4.5,5.5],weights=[284,467,189,56,3,1],edgecolor='black', linewidth=1.2,color='skyblue')
plt.hist([8,9,10,11,12,13,14,15],bins=[7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5],weights=[5,40,149,448,211,128,16,3],edgecolor='black', linewidth=1.2,color='lightgreen')
plt.hist([19,20,21,22,23],bins=[18.5,19.5,20.5,21.5,22.5,23.5],weights=[56,225,563,127,29],edgecolor='black', linewidth=1.2,color='yellow')
plt.hist([28,29,30,31],bins=[27.5,28.5,29.5,30.5,31.5],weights=[1,57,202,740],edgecolor='black', linewidth=1.2,color='orange')
plt.ylim(0,800)
plt.legend(('Vin=0 W=0','Vin=5 W=15','Vin=15 W=10','Vin=15 W=15'),loc='upper left')
plt.savefig('MC.png')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:35:30 2019

@author: skodge
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
'''


y_bl=np.array([[31,	29,	27,	25,	23,	21,	19,	16,	19,	16,	15,	12,	11,	8,	5,	1],
[29,	27,	26,	23,	23,	20,	18,	15,	17,	15,	14,	11,	10,	7,	5,	1],
[28,	26,	25,	22,	21,	19,	17,	15,	16,	15,	13,	11,	9,	7,	4,	1],
[27,	25,	23,	21,	20,	17,	16,	15,	15,	14,	12,	10,	9,	6,	4,	1],
[25,	23,	21,	19,	18,	16,	15,	14,	15,	13,	11,	9,	8,	5,	3,	1],
[23,	21,	19,	17,	16,	15,	15,	13,	13,	11,	10,	8,	7,	5,	3,	1],
[20,	18,	17,	15,	15,	15,	13,	11,	12,	10,	9,	7,	6,	4,	3,	1],
[17,	16,	15,	15,	14, 13, 12, 10,	11,	9,	8,	6,	5,	4,	3,	1],
[15,	15,	14,	13,	12,	11,	10,	9,	9,	8,	7,	5,	5,	3,	2,	1],
[14,	13,	12,	11,	10,	9,	9,	7,	8,	7,	6,	5,	4,	3,	2,	1],
[12,	11,	10,	9,	9,	8,	7,	6,	6,	5,	5,	4,	3,	2,	1,	1],
[9,	9,	8,	7,	7,	6,	5,	5,	5,	4,	4,	3,	3,	2,	1,	1],
[7,	7,	6,	5,	5,	4,	4,	3,	3,	3,	3,	2,	2,	1,	1,	0],
[5,	5,	4,	3,	3,	3,	3,	3,	3,	2,	2,	1,	1,	1,	1,	0],
[3,	3,	3,	2,	2,	2,	2,	2,	2,	1,	1,	1,	1,	1,	0,	0],
[2,	2,	2,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0]])
Vin=np.array([15, 14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
W=np.array([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
plt.plot(Vin,y_bl)
plt.ylabel('Digital Output')
plt.xlabel('Vin')
plt.legend(('W=15','W=14','W=13','W=12','W=11','W=10','W=9','W=8','W=7','W=6','W=5','W=4','W=3','W=2','W=1','W=0'))


plt.figure()
plt.plot(W,y_bl.T)
plt.ylabel('Digital Output')
plt.xlabel('2nd Operand')
plt.legend(('Vin=15','Vin=14','Vin=13','Vin=12','Vin=11','Vin=10','Vin=9','Vin=8','Vin=7','Vin=6','Vin=5','Vin=4','Vin=3','Vin=2','Vin=1','Vin=0'))


product= (Vin.reshape(16,1)*W.reshape(1,16))*31/225
product= np.round(product.reshape(256,))
plt.figure()
plt.plot(product,product)
plt.scatter(product,y_bl.reshape(256,))
'''
matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
y_bl=np.array([[31,	29,	27,	25,	23,	21,	19,	16,	19,	16,	15,	12,	11,	8,	5,	1],
[23,	21,	19,	17,	16,	15,	15,	13,	13,	11,	10,	8,	7,	5,	3,	1],
[12,	11,	10,	9,	9,	8,	7,	6,	6,	5,	5,	4,	3,	2,	1,	1],
[2,	2,	2,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0]])
W=np.array([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
plt.ylabel('Digital Output')
plt.xlabel('W')
plt.plot(W,y_bl.T,'-o')
plt.legend(('Vin=15','Vin=10','Vin=5','Vin=0'))
plt.savefig('OutVsW.png')

plt.figure(figsize=(10,5))
y_bl=np.array([[31,	21,	15,	1],
[29,	20,	14,	1],
[28,	19,	13,	1],
[27,	17,	12,	1],
[25,	16,	11,	1],
[23,	15,	10,	1],
[20,	15,	9, 1],
[17,	13,	8,	1],
[15,	11,	7,	1],
[14,	9,	6,	1],
[12,	8,	5,	1],
[9,	6,	4,	1],
[7,	4,	3,	0],
[5,	3,	2,	0],
[3,	2,	1,	0],
[2,	1,	1,	0]])
Vin=np.array([15, 14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
plt.ylabel('Digital Output')
plt.xlabel('Vin')
plt.plot(Vin,y_bl,'-o')
plt.legend(('W=15','W=10','W=5','W=0'))
plt.savefig('OutVsVin.png')

plt.figure(figsize=(10,5))
y_bl=np.array([[31,	29,	27,	25,	23,	21,	19,	16,	19,	16,	15,	12,	11,	8,	5,	1],
[29,	27,	26,	23,	23,	20,	18,	15,	17,	15,	14,	11,	10,	7,	5,	1],
[28,	26,	25,	22,	21,	19,	17,	15,	16,	15,	13,	11,	9,	7,	4,	1],
[27,	25,	23,	21,	20,	17,	16,	15,	15,	14,	12,	10,	9,	6,	4,	1],
[25,	23,	21,	19,	18,	16,	15,	14,	15,	13,	11,	9,	8,	5,	3,	1],
[23,	21,	19,	17,	16,	15,	15,	13,	13,	11,	10,	8,	7,	5,	3,	1],
[20,	18,	17,	15,	15,	15,	13,	11,	12,	10,	9,	7,	6,	4,	3,	1],
[17,	16,	15,	15,	14, 13, 12, 10,	11,	9,	8,	6,	5,	4,	3,	1],
[15,	15,	14,	13,	12,	11,	10,	9,	9,	8,	7,	5,	5,	3,	2,	1],
[14,	13,	12,	11,	10,	9,	9,	7,	8,	7,	6,	5,	4,	3,	2,	1],
[12,	11,	10,	9,	9,	8,	7,	6,	6,	5,	5,	4,	3,	2,	1,	1],
[9,	9,	8,	7,	7,	6,	5,	5,	5,	4,	4,	3,	3,	2,	1,	1],
[7,	7,	6,	5,	5,	4,	4,	3,	3,	3,	3,	2,	2,	1,	1,	0],
[5,	5,	4,	3,	3,	3,	3,	3,	3,	2,	2,	1,	1,	1,	1,	0],
[3,	3,	3,	2,	2,	2,	2,	2,	2,	1,	1,	1,	1,	1,	0,	0],
[2,	2,	2,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0]])
Vin=np.array([15, 14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
W=np.array([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
product= (Vin.reshape(16,1)*W.reshape(1,16))*31/225
product= np.round(product.reshape(256,))
plt.ylabel('Digital Output')
plt.xlabel('Vin*W')
plt.plot(product,product)
plt.scatter(product,y_bl.reshape(256,),marker="x",c='red')
plt.legend(('Expected','Observed'))
plt.savefig('ExpVsObs.png')

plt.figure(figsize=(10,6))
matplotlib.rcParams.update({'font.size': 20})
plt.ylabel('Number of Samples')
plt.xlabel('Digital Output')
plt.hist([0,1,2,3,4,5],bins=[-.5,.5,1.5,2.5,3.5,4.5,5.5],weights=[284,467,189,56,3,1],edgecolor='black', linewidth=1.2,color='skyblue')
plt.hist([8,9,10,11,12,13,14,15],bins=[7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5],weights=[5,40,149,448,211,128,16,3],edgecolor='black', linewidth=1.2,color='lightgreen')
plt.hist([19,20,21,22,23],bins=[18.5,19.5,20.5,21.5,22.5,23.5],weights=[56,225,563,127,29],edgecolor='black', linewidth=1.2,color='yellow')
plt.hist([28,29,30,31],bins=[27.5,28.5,29.5,30.5,31.5],weights=[1,57,202,740],edgecolor='black', linewidth=1.2,color='orange')
plt.ylim(0,800)
plt.legend(('Vin=0 W=0','Vin=5 W=15','Vin=15 W=10','Vin=15 W=15'),loc='upper left')
plt.savefig('MC.png')


