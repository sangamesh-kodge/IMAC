#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:55:12 2020

@author: skodge
"""
from system_analysis import system_analysis
'''
Compute primitive 
delay['comp'] : time for 1 cycle of MAC operation
delay['adc'] : time for ADC operation per cycle of compute 
delay['dac'] : time for DAC operation per cycle of compute

energy['comp'] : energy for MAC operation/size of MAC 
energy['adc'] : energy for ADC operation per cycle of compute 
energy['dac'] : energy for DAC operation per cycle of compute
Refer to table in IMAC paper for details about other parameters
'''

delay = {'comp' : 1.0,
         'adc'  : 5.0/10,
         'dac'  : 0.0,
         'mult' : 4.0,
         'read' : 4.0}
energy = {'comp' : 0.2798224,
          'adc'  : 0.1,
          'dac'  : 0.0,
          'mult' : 0.9,
          'read' : 5.2,
          'reg'  : 4,
          'Pleak': 0.0024}
arch_hard = {'Bio'  : 16,
             'Bw'   : 5,
             'Ncol' : 256,
             'Nrow' : 256,
             'Nmult': 175,
             'Nbank': 4}
arch_soft = {'K' : [5, 5, 1, 1, 1],
             'M' : [1, 6, 400, 120, 84],
             'N' : [6, 16, 120, 84, 10],
             'L' : [28,14,1,1,1],
             'P' : [2,0,0,0,0],
             'S' : [1,1,1,1,1]}

params = {'delay'    : delay,
         'energy'   : energy,
         'arch_hard': arch_hard,
         'arch_soft': arch_soft}
IMAC = system_analysis (**params)
Tvn, T, Evn, E = IMAC.compute()
print ("Tvn : {} Evn : {} EDPvn : {} \n  T : {} E : {} EDP : {}".format(Tvn,Evn,Tvn*Evn, T, E, T*E))
     