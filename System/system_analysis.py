#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:19:26 2020

@author: skodge
"""

class system_analysis():
    def __init__(self, **params):
        self.delay = params['delay']
        self.energy = params['energy']
        self.arch_hard = params['arch_hard']
        self.arch_soft = params['arch_soft'] 
           
    def delay_compute (self, K, M, N, L, P, S):
        T = 0 
        Tvn = 0
        for i in range(len(L)):
            Nmov = (L[i] + P[i] * 2 - K [i]) / (S[i]) + 1
            T += M[i] * N[i] * K[i]**2  * Nmov**2 *  (self.arch_hard['Bw'] / (self.arch_hard['Ncol'] * self.arch_hard['Nbank'] )) * (self.delay['comp'] + self.delay['adc'] + self.delay['dac']  )
            Tvn += M[i] * N[i] * K[i]**2 * (self.arch_hard['Bw']/ (self.arch_hard['Bio'] * self.arch_hard['Nbank'])) * self.delay['read'] + (M[i] * N[i] * K[i]**2 * Nmov**2 / self.arch_hard['Nmult']) * self.delay['mult']  
        return Tvn, T
    
    def energy_compute (self, K, M, N, L, P, S):
        E = 0 
        Evn = 0
        for i in range(len(L)):
            Nmov = (L[i] + P[i] * 2 - K [i]) / (S[i]) + 1
            E += M[i] * N[i] * K[i]**2  * Nmov**2 * (self.energy['comp'] + self.energy['adc'] + self.energy['dac'])
            Evn += M[i] * N[i] * K[i]**2 * self.energy['read'] + M[i] * N[i] * K[i]**2 * Nmov**2 * self.energy['mult'] + M[i] * N[i] * Nmov**2 *self.energy['reg']  
        return Evn, E
    
    def compute(self):
        Tvn, T = self.delay_compute(K = self.arch_soft['K'], M = self.arch_soft['M'], N = self.arch_soft['N'], L = self.arch_soft['L'], P = self.arch_soft['P'], S = self.arch_soft['S'])
        Evn, E = self.energy_compute(K = self.arch_soft['K'], M = self.arch_soft['M'], N = self.arch_soft['N'], L = self.arch_soft['L'], P = self.arch_soft['P'], S = self.arch_soft['S'])
        Evn, E = Evn + self.energy['Pleak'] * Tvn * 1e-9, E + self.energy['Pleak'] * T * 1e-9 
        return Tvn, T, Evn, E
        
        
        