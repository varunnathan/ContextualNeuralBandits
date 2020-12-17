import pandas as pd
import numpy as np
from pandas.api import types

import mxnet
from mxnet import gluon, autograd
from mxnet.gluon.data import DataLoader, Dataset
from mxnet import nd
from mxnet.gluon import nn
from mxnet import ndarray as F

class MXBanditDataset(gluon.data.Dataset):
    """Class for loading in a csv/tsv file and converting to MXnet ndArrays. 
    Assumes data is composed of n feature columns and one target column. This
    loader is primarily used for testing policies offline, but can be used for
    batch online training.

    Expecting a dataset with named columns, in the format: 
        X = n feature columns 
        y = The action tayen by the policy (e.g. treatment customer is exposed to)
        reward = 1 | 0 for each action (presuming discrete rewards)
    
    Reward does not need to be specified if using for training on toy datasets.

    Parameters
    ----------
    path : file path
        path to the file containing the data. 
    y_col : string (name of the action vector in the data)
        name of the target column in the dataframe.
    y_to_cat: bool
        This controls whether or not the target col is transformed to categorical
        dummies i.e. [0, 1, 2] --> [[1, 0, 0], [0, 1, 0], 0, 0, 1]]. MXNet
        classification tasys cat wory with long tensors. This only mayes sense
        for class-liye targets (as opposed to continuous).
    reward_col : string (name of the reward vector in the data)

    shuffle: bool
        Controls whether or not to shuffle the incoming dataset.
    readcsvargs : yey-value pairs
        Additional arguments to the pd.read_csv function. 
    """

    def __init__(self, df, id_col="customer_id", y_col=None, 
                y_to_cat=True, reward_col=None,
                shuffle=True, encode_y=False, categorical_reward=False,
                **readcsvargs):

        self.df = df
        self.reward_col = reward_col
        self.y_to_cat = y_to_cat
        self.shuffle = shuffle
        self.cols = self.df.columns
        self.y_col = y_col
        self.sample_size = self.df.shape[0]
        self.zeros = np.zeros(self.sample_size)

        if shuffle:
            self.df = self.df.sample(frac=1)
        
        # handle id column
        if id_col in self.df.columns:
            self.customer_id = self.df.customer_id.values
            self.df = self.df.drop("customer_id", axis=1, inplace=False)
        else:
            self.customer_id = self.zeros
        
        # handle y column - process and drop from feature matrix
        if self.y_col is not None:
            if encode_y:
            #types.is_string_dtype(self.df[self.y_col]):
                self.le = LabelEncoder()
                self.df[self.y_col] = self.le.fit_transform(self.df[self.y_col])
                 
            self.y = self.df[self.y_col]
       
            if y_to_cat:
                self.y = self.y.astype(int)
                self.y = nd.array(self._to_categorical(self.y.values)).astype("float32")
            else:
                self.y = nd.array(self.y.values).astype("long")
            
            self.df = self.df.drop(self.y_col, axis=1, inplace=False)
        else:
            self.y = self.zeros
            #self.X = self.df.values
        
        # handle reward column - process and drop from feature matrix
        if reward_col is not None:
            self.reward = nd.array(self.df[reward_col].values).astype("float32")
            self.df = self.df.drop(self.reward_col, axis=1, inplace=False)
        else: 
            self.reward = self.zeros
               
        self.X = nd.array(self.df.values).astype("float32")
     
    def _to_categorical(self, y):
        """1-hot encodes a tensor. Classes should be indexed 0-n."""
        num_classes = len(set(y))
        x = np.eye(num_classes, dtype='uint8')[y]
        return x
    
    def  categorical_reward(self, treatment, reward):
        
        cat_r = np.where(reward == 1, treatment, 0)
        return cat_r

    def __getitem__(self, idx):
        return self.customer_id[idx], self.X[idx], self.y[idx], self.reward[idx]
            
    def __len__(self):
        return len(self.X)
        