# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:05:17 2021

@author: Lu, Yihe

Original Minverva2
"""

import numpy as np

class Minerva2:
    def __init__(self, trace_size):
        self.trace_size = trace_size
        self.reset()
    
    def reset(self):
        self.memory = np.empty((0, self.trace_size))

    def learn(learning_data, learning_rate = 1):
        _data = canonicalisation(learning_data, self.trace_size)
        self.memory = np.concatenate((self.memory, _data), axis = 0)
        
    def respond():