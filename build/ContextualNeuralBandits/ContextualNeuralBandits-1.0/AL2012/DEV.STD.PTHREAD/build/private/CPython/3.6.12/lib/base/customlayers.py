import mxnet
from mxnet import gluon, autograd
from mxnet import nd
from mxnet.gluon import nn
from mxnet import ndarray as F
import numpy as np

class ConcreteDropout(gluon.Block):
    """This module allows to learn the dropout probability for any given input layer.
    Parameters
    ----------
        layer: a layer Module.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (tau * N)$
            with prior lengthscale l, model precision tau (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                 dropout_regularizer = 2 / (tau * N)$
            with model precision tau (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                 weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """
    def __init__(self, layer, input_shape, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.3, init_max=.5):
        super().__init__()
        with self.name_scope():
            # Post drop out layer
            self.layer = layer
            # Input dim for regularisation scaling
            self.input_dim = np.prod(input_shape[1:])
            # Regularisation hyper-parameters
            self.weight_regularizer = weight_regularizer
            self.dropout_regularizer = dropout_regularizer
            # Initialise p_logit
            init_min = np.log(init_min) - np.log(1. - init_min)
            init_max = np.log(init_max) - np.log(1. - init_max)
            rnd = np.random.uniform(init_min, init_max)

            self.p_logit = self.params.get(
                "p_logit", init=mxnet.init.Constant(rnd), shape=1
            )
            self.all_p = []
            self.reg_all = []

    def forward(self, x):
        out = self.layer(self._concrete_dropout(x))
        return out 

    def regularisation(self):
        """Computes weights and dropout regularisation for the layer, has to be
        extracted for each layer within the model and added to the total loss
        """
        with autograd.record():
            weights_regularizer = self.weight_regularizer * self._sum_n_square() / (1 - self.p)
            dropout_regularizer = self.p * nd.log(self.p)
            dropout_regularizer = dropout_regularizer + (1. - self.p) * nd.log(1. - self.p)
            dropout_regularizer = dropout_regularizer * self.dropout_regularizer * self.input_dim
            regularizer = weights_regularizer + dropout_regularizer
        self.reg_all.append(regularizer)
        return regularizer

    def _concrete_dropout(self, x):
        """Forward pass for dropout layer
        """
        with autograd.record():
            eps = 1e-7
            temp = 0.1
            
            self.p = nd.sigmoid(self.p_logit.data())

            # Check if batch size is the same as unif_noise, if not take care
            unif_noise = nd.array(np.random.uniform(size=tuple(x.shape)))

            drop_prob = (nd.log(self.p + eps)
                        - nd.log(1 - self.p + eps)
                        + nd.log(unif_noise + eps)
                        - nd.log(1 - unif_noise + eps))
            drop_prob = nd.sigmoid(drop_prob / temp)
            random_tensor = 1 - drop_prob
            retain_prob = 1 - self.p
            x  = nd.multiply(x, random_tensor)
            x = x / retain_prob
        self.all_p.append(self.p)
        return x

    def _sum_n_square(self):
        """Helper function for paramater regularisation
        """
        sum_of_square = 0
        pdict = self.layer.params
        with autograd.record():
            for param in pdict:
                sum_of_square = sum_of_square + nd.sum(nd.power(pdict[param].data(), 2))
        return sum_of_square
    