import mxnet
from mxnet import gluon, autograd
from mxnet import nd
from mxnet.gluon import nn
from mxnet import ndarray as F
from src.base.base_neural_bandit import BaseDropoutBandit

def SequentialLinear(*layer):
    l = nn.Sequential()
    with l.name_scope():
        l.add(layer)
    return l

class MultitaskBandit(BaseDropoutBandit):
    """
    This is the main class for building Contextual Neural Bandits. The structure of the network is 
    Multitask. Each action (e.g. marketing content selection) being represented by a distinct task that is learned by the network. 
    Specifically, each decision is represented by a distinct output layer, 
    with all output layers being connected to a shared hidden layers. 
    This enables the network to learn specific features that are associated with each action, 
    whilst also benefitting from shared features learned in the hidden layers. 
    This build on the neural network committee approach, but is more efficient 
    as only one network is used and also enables information sharing across the various tasks. 

    Parameters:
    -----------
        input_n : The number of input features
        hidden_n : The number of hidden units in each layer
        actions_n : The number of actions (e.g. content decisions) the bandit is making
        layers_n : The number of layers in the network
        dropout_p : The dropout rate for layers in the model 
        layer_szs : When using multiple layers you can specify each layer's shape by passing a list of sizes e.g. [100, 50, 20]
                    If None, each layer will have hidden_n units 
        act_out : activation for the output layer
        sm_out : Apply softmax to the output layer
        log_sm_out : Apply log_softmax to the output layer
    """
  
    def __init__(self, input_n, hidden_n, actions_n, layers_n=1,
     layer_szs=None, act_layers="relu", act_out="sigmoid", sm_out=False, 
     log_sm_out=False, dropout_p=0.5, **kwargs):
        super().__init__(input_n, hidden_n, actions_n, layers_n=layers_n,
     layer_szs=layer_szs, act_out=act_out, sm_out=sm_out, 
     log_sm_out=log_sm_out, dropout_p=dropout_p, **kwargs)

        with self.name_scope():
            self.input_n    = input_n
            self.hidden_n   = hidden_n
            self.actions_n  = actions_n
            self.dropout_p  = dropout_p
            self.sm_out     = sm_out
            self.log_sm_out = log_sm_out
            self.act_layers = act_layers
            self.act_out    = act_out
            self.layers_n = layers_n
            if layer_szs is None:
                self.layer_szs   = [self.hidden_n] * self.layers_n
            else:
                self.layer_szs   = layer_szs
            self.selections = list(range(actions_n))
            # model architecture            
                
            self.hidden = nn.Sequential()
            with self.hidden.name_scope():
                if len(self.layer_szs) > 1:
                    for i in range(self.layers_n):
                        self.hidden.add(nn.Dense(units=self.layer_szs[i], activation=self.act_layers), nn.Dropout(self.dropout_p))
                else:
                     for i in range(self.layers_n):
                        self.hidden.add(nn.Dense(units=self.hidden_n, activation=self.act_layers), nn.Dropout(self.dropout_p))
                    
            self.actions = nn.Sequential()
            with self.actions.name_scope():
                for i in range(self.actions_n):
                    self.actions.add(nn.Dense(in_units=self.layer_szs[-1], units=2, activation=self.act_out))      
                    
    def forward(self, context):
        
        shared = self.hidden(context)
        if self.sm_out:
            preds = nd.softmax(nd.stack(*[l(shared)[:, 1] for l in self.actions]).T, axis=1)
        elif self.log_sm_out:
            preds = nd.log_softmax(nd.stack(*[l(shared)[:, 1] for l in self.actions]).T, axis=1)
        else:
            preds = nd.stack(*[l(shared)[:, 1] for l in self.actions]).T
            
        return preds
    
    def __str__(self):
        return "MultitaskBandit"