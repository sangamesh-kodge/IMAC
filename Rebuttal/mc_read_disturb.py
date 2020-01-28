#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 07:03:59 2019

@author: skodge
"""

import numpy as np
a = np.loadtxt("MC_read_disturb.txt").T


if(a[1].max()>a[4].min()):
    print ('Cell storing one flipped')
else:
    if(a[2].max()>a[7].min()):
        print ('Cell storing zero flipped')
    else:
        print ('None of the cells flipped')
if (a[1].max()>a[2].max()):
    print (a[1].max()*1000)
else:
    print (a[2].max()*1000)

if (a[4].max()>a[7].max()):
    print (1000-a[7].max()*1000)
else:
    print (1000-a[4].max()*1000)
        
    
    