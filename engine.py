import numpy as np
import random

class Value:
    def __init__(self,data,_children=(),_op = '',label=''):
        self.data = np.float64(data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda : None
        self.grad = np.float64(0.0)
        self.label = label

    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad += 1.0*out.grad;
            other.grad += 1.0*out.grad
        out._backward = _backward
        return out
        
    def __neg__(self):
        return self*-1

    def __sub__(self,other):
        return self + (-other)
    def __rsub__(self,other):
        return other + (-self)
    def __radd__(self,other):
        return self+other
        
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)  
        out = Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data
        out._backward = _backward
        return out
    def __rmul__(self,other):
        return self*other

    def __truediv__(self,other):
        return self * other**-1

    def __pow__(self,other):
        assert isinstance(other,(int,float)) ,"Only Support int/float for Power"
        out = Value(self.data**other ,(self,),label = f'**{other}')
        def _backward():
            self.grad += other*(self.data**(other-1))*out.grad
        out._backward = _backward
        return out
        
    def sigmoid(self):
        x = self.data
        sig_val = 1.0/(1.0+np.exp(-x)) if x>=0 else  np.exp(x)/(1.0 + np.exp(x))
        out = Value(sig_val,(self,),'Sigmoid')
        def _backward():
            self.grad += out.grad*sig_val*(1-sig_val)
        out._backward = _backward
        return out
        
    def relu(self):
        out = Value(0 if(self.data < 0) else self.data ,(self,),"Relu" )
        def _backward():
            self.grad += (out.data>0)*out.grad
        out._backward = _backward
        return out
        
    def tanh(self):
        n = self.data
        x = (np.exp(2*n) - 1) / (np.exp(2*n) + 1)
        out = Value(x,(self,),'tanh','tanh')
        def _backward():
            self.grad += out.grad*(1-x**2)
        out._backward = _backward
        return out
        
        
    def exp(self):
        out = Value(np.exp(self.data),(self,),label = 'exp')
        def _backward():
            self.grad += out.grad*out.data
        out._backward = _backward 
        return out
        
    def backward(self):
        nodes = []
        edge = set()
        def visit(Value):
            if Value not in edge:
                edge.add(Value)
                for child in Value._prev:
                    visit(child)
                nodes.append(Value)
        visit(self)
        nodes.reverse()
        #print(nodes)
        self.grad = 1.0
        for Value in nodes:
            Value._backward()
            
    def __repr__(self):
        return f"Value(data = {self.data})"
