import mxnet
from mxnet import gluon, autograd
from mxnet import nd
from mxnet.gluon import nn
from mxnet import ndarray as F
import numpy as np

class BaseDropoutBandit(gluon.Block):

    def __init__(self, input_n, hidden_n, actions_n, layers_n=1,
     layer_szs=None, act_out="sigmoid", sm_out=False, 
     log_sm_out=False, dropout_p=0.5):

        super().__init__()
        with self.name_scope():
            self.criterion  = None
            self.optimizer  = None
            self.trainer    = None
            self.mode       = None
            self.modes      = {
                "train" : autograd.train_mode(), 
                "predict" : autograd.predict_mode()
            }
    
    def dropout_action_choice(self, context, with_dropout=True):
        """Choose best action from action predictions. Used in conjunction with
        dropout predictions this approximates Thompson sampling."""

        context = self.check_input(context)

        # when model is in train mode dropout is used when making predictions
        _mode = "train" if with_dropout else "predict"
        
        self.mode = self.modes[_mode]
        with autograd.record():
            with self.mode:
                yhat = self.forward(context)
            # choose "best" treatment and associated prediction
            action_pred = nd.max(yhat, axis=1)
            action = nd.argmax(yhat, axis=1).astype("long") 
            
        return action_pred, action

    def greedy_action_choice(self, context, e=0.10, with_dropout=False):
        """Sample actions randomly with e probability. Exploit with 1-e probability."""

        context      = self.check_input(context)
        self.batch_size   = context.shape[0]
        
        _mode = "train" if with_dropout else "predict"
        self.mode = self.modes[_mode]
        
        take_greedy_choice = nd.array(np.random.uniform(0, 1, size=self.batch_size)) > e    
        rnd_selections = nd.array(np.random.choice(self.selections, size=self.batch_size, replace=True))
        
        with autograd.record():
            with self.mode:
                yhat = self.forward(context)
                greedy_selections = nd.argmax(yhat, 1)
                
                actions = nd.where(take_greedy_choice, greedy_selections, rnd_selections).astype("long")
                preds = yhat.pick(actions)

        return preds, actions

    def update(self, action_pred, reward, step=False, weight=1., batch_size=None):
        """Bandit algos only take into account the loss corresponding to 
        the prediction for the observed action.
        Parameters
        ----------
        action_pred : nd.array (float)
            The prediction associated with the best action.
        reward : nd.array (float)
            The reward generated by the policy. 
        step : bool
            Whether or not to backpropogate the gradients after each update 
        weight : nd.array(float)
            Sample weights for balanced training. len(weight)==len(reward)
            If not set, all samples are factored equally.
        """
        
        if batch_size is None:
            batch_size = action_pred.shape[0]

        with autograd.record():
            if str(self) == 'ConcreteDropoutBandit':
                reg = self.regularisation_loss()
            else:
                reg = nd.array([0.])
             # loss only considers prediction for the chosen action.   
            loss = self.criterion(action_pred, reward) 
        loss.backward()
        if step:
            self.trainer.step(batch_size) 
        else:
            Warning("Note: step is set to False and, therefore, calculated gradients are not backpropagating")
        return loss.asnumpy()

    def xavier_init(self, rnd_type="gaussian"):
        self.collect_params().initialize(mxnet.init.Xavier(rnd_type=rnd_type))

    def check_input(self, context):
        if not isinstance(context, np.float64):
            Warning("Converting to gluon")
            context = context.astype("float32")

        if context.ndim == 1:
            context = context.shape(1, -1)
        
        return context
    
    def _greedy_indexing(self, yhat, e):
        
        take_greedy_choice = nd.array(np.random.uniform(0, 1, size=self.batch_size)) > e
        rnd_selections = nd.array(np.random.choice(self.selections, size=self.batch_size, replace=True))
        greedy_selections = nd.argmax(yhat, 1)
    
        with autograd.record():
            greedy_actions = nd.where(take_greedy_choice, greedy_selections, rnd_selections)
            greedy_preds = yhat.pick(greedy_actions)

        return greedy_preds, greedy_actions
    
    def set_gradients_to_accumulate(self):
        """Sets gradients to accumulate rather than to reset after each gradient calculation."""
        for p in self.collect_params().values():
            p.grad_req = 'add'
