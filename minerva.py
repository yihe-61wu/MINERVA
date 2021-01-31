# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:05:17 2021

@author: Lu, Yihe

Original Minverva2 from Hintzman (1984)
"""

import numpy as np

def proba0(x):
    _x = np.asarray(x)
    _x[_x > 1], _x[_x < -1] = 1, -1
    p = 1 - np.abs(_x)
    return p

class Minerva2: # Hintzman (1984)
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
    
    def _probe_similarity(self, probe): # Eq.1
        numer = np.dot(self.memory, probe)
        denom = self.trace_size - np.dot(proba0(self.memory), proba0(probe))
        if denom == 0:
            sim = np.zeros(len(probe))
        else:
            sim = np.divide(numer, denom)
        return sim