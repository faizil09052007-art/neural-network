import numpy as np
import random
from engine import Value

class Neuron:
    def __init__(self,nin):
        self.w = np.array([Value(random.uniform(-1,1)) for _ in range(nin)])
        self.b = Value(random.uniform(-1,1))
    def parameters(self):
        p = np.append(self.w, self.b)
        return p
    
    def __call__(self,x):
        act = (self.w*x).sum()  + self.b
        out = act.sigmoid()
        return out
        
class Layer:
    def __init__(self,nin,nout):
        self.neurons  = np.array([Neuron(nin) for _ in range(nout)])

    def parameters(self):
        parms = []
        for neuron in self.neurons:
            parms.extend(neuron.parameters())
        return parms
                
    def __call__(self,x):
        out = np.array([n(x) for n in self.neurons])
        return out

class MLP:
    def __init__(self,nin,nout):
        sz = [nin] + nout
        self.layers = np.array([Layer(sz[i],sz[i+1] ) for i in range(len(nout))])
    def parameters(self):
        parms = []
        for layer in self.layers:
            parms.extend(layer.parameters())
        return parms
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return  x[0] if len(x)==1 else x 
