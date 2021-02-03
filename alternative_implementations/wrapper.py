import numpy as np

import minerva2 as dwhite54

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
    
    def _echo(self, probs):
        