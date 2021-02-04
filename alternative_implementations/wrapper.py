import numpy as np

import minerva2 as dwhite54
import rpy2.robjects as robjects

class model1(dwhite54.Minerva2):
    def reset(self):
        self.__init__(self.features_per_trace)
        
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
    def __init__(self, feature_num):
        deniztu = robjects.r['source']
        deniztu("minerva-al.R")
        self.r_probe_memory = robjects.globalenv['probe_memory']
        self.r_expect_event = robjects.globalenv['expect_event']
        self.r_learn = robjects.globalenv['learn']
        ###
        self.feature_num = feature_num
        self.reset()
    
    def reset(self):
        self.memory = np.empty((0, self.trace_size))
        
    def learn(self, learning_data):
        pass