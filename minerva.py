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
    
    def _echo(self, probes):
        _probe = np.reshape(probes, (-1, self.trace_size))
        similarity = np.dot(self.memory, _probe.T) # Eq. 1
        activation = similarity ** 3 # Eq. 2
        intensity = np.sum(activation, axis = 0) # Eq. 3
        content = np.dot(activation.T, self.memory) # Eq. 4
        normalised_echo = content / np.amax(np.abs(content), axis = 1).reshape((-1, 1))
        return intensity, normalised_echo
        
    
    
    
    
    
    
    
    
    
    