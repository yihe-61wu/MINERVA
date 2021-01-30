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

    def learn(self, learning_data, learning_rate = 1):
        _data = np.reshape(learning_data, (-1, self.trace_size))
        self.memory = np.concatenate((self.memory, _data), axis = 0)
        
    def respond(self, probes, recurrence = 1):
        pass
    
    def _probe_similarity(self, probe):
        numer = np.dot(self.memory, probe)
        denom = 0
        return np.divide(numer, denom)