import numpy as np

import alternative_implementations.minerva2 as dwhite54
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects.numpy2ri as rpyn

class model1(dwhite54.Minerva2):
    def reset(self):
        self.__init__(self.features_per_trace)
        
    def get_memory_matrix(self):
        return self.model
        
    def learn(self, learning_data):
        self.add_traces(np.reshape(learning_data, (-1, self.features_per_trace)), 0)
    
    def respond(self, probes, recurrence = 1):
        echo = probes[:]
        for epoch in range(recurrence):
            echo = self._echo(echo)[1]
        return echo
    
    def _echo(self, probes):
        intensity, activation = self.get_echo_intensities(np.reshape(probs, (-1, self.features_per_trace)), 0)        
        content = np.dot(activation, self.model)
        normalised_echo = content / np.amax(np.abs(content), axis = 1).reshape((-1, 1))
        return intensity, normalised_echo

class model2:
    def __init__(self, trace_size):
        file = open('alternative_implementations/minerva-al.R')
        string = ''.join(file.readlines())
        self.funs = SignatureTranslatedAnonymousPackage(string, 'functions')
        # ###
        self.model = 'Minerva2'
        self.trace_size = trace_size
        self.reset()
    
    def reset(self):
        self.memory = np.empty((0, self.trace_size))
        
    def get_memory_matrix(self):
        return self.memory
        
    def learn(self, learning_data):
        for row in learning_data:
            past_memory, new_event = [self._py2ri(data) for data in [self.memory, row]]
            new_memory = self.funs.learn(event = new_event, memory = past_memory, p_encode = 1, model = self.model) 
            self.memory = self._ri2py(new_memory)
    
    def respond(self, probes, recurrence = 1):
        echo = probes[:]
        for epoch in range(recurrence):
            echo = self._echo(echo)
        return echo        
    
    def _echo(self, probes):
        echo = []
        for row in probes:
            past_memory, new_probe, cueidx = [self._py2ri(data) for data in [self.memory, [row], np.arange(self.trace_size) + 1]]
            r_echo = self.funs.probe_memory(probe = new_probe, memory = past_memory, cue_feature = cueidx, model = self.model)
            echo.append(r_echo)
        return np.asarray(echo)
    
    def _py2ri(self, data):
        return rpyn.py2ri(np.asarray(data))
    
    def _ri2py(self, data):
        return rpyn.ri2py(data)