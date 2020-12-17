import pandas as pd
import numpy as np
import pandas as pd
import mxnet
from mxnet import gluon, autograd
from mxnet.gluon.data import DataLoader, Dataset
from mxnet import nd
from mxnet.gluon import nn
from mxnet import ndarray as F
from sklearn.utils import compute_sample_weight
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns

def mx_compute_sample_weight(y, class_weight={1 : 1., 0 : .7}):
    """Compute sample weights for more balanced training. For Bandits this is
    applied to the reward. That is, samples that result in rewards are weighted
    more highly than no reward samples."""

    if isinstance(y, mxnet.ndarray.ndarray.NDArray):
        y = y.asnumpy()
    
    try:
        weights = compute_sample_weight(class_weight=class_weight, y=y)
    except ValueError as e:
        value = int(np.unique(y)[0])
        weights = np.repeat(y.shape[0], class_weight[value])

    return nd.array(weights)

def plot_k_choices(choices, title="Choice plot"):
    plt.figure(figsize=(20,8))
    x = np.arange(0, len(choices))
    plt.scatter(x, choices, c=choices, marker=".", alpha=1)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Trial", fontsize=16, fontweight="bold")
    plt.ylabel("Variant", fontsize=16, fontweight="bold")
    plt.yticks(list(range(4)))
    plt.colorbar();

def plot_reward_bound(regret, batch_size, scaler=1, labels=None, **kwargs):
    
    plt.figure(figsize=(12, 6))
    if type(regret) == list:
        labels = ("one", "two") if labels is None else labels
        for ix, i in enumerate(regret):
            meangret = pd.Series(i).rolling(window=10).mean()
            sdre = i.std() * scaler
            plt.plot(meangret, linewidth=3.0, alpha=.4, label=labels[ix], **kwargs)
            plt.fill_between(list(range(len(i))), meangret+sdre, 
                            np.clip(meangret-sdre, 0, max(i)),
                                alpha=.2)
            plt.xlabel("Batch")
            plt.ylabel("Rewards sum per batch".format(batch_size))
            plt.legend();
    else:
        meangret = pd.Series(regret).rolling(window=10).mean()
        sdre = regret.std() * scaler
        plt.plot(meangret, linewidth=3.0, alpha=.4, **kwargs)
        plt.fill_between(list(range(len(regret))), meangret+sdre, 
                            np.clip(meangret-sdre, 0, max(regret)),
                                alpha=.2);
        plt.xlabel("Batch")
        plt.ylabel("Rewards sum per batch".format(batch_size));

def namedtuple_to_dataframe(log):
    field_names = log._fields
    values = []
    for field in field_names:
        value  = getattr(log, field)
        values.append(value)
    df = pd.DataFrame([values], columns=field_names)
    return df