import json
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import os
import src.util.data_handler as dh
from src.util.const import UNIT, FIGURE_PATH
from src.util.naming import *
import seaborn as sns

_FIGURE_PATH = os.path.join(os.getcwd(), 'dat/figures/')

def plot():
    fig, ax = plt.subplots()
    sns.lmplot(x='Node Degree', y='Cover', hue='Strategy', data=df, markers=_STRATEGY_MARKERS)
    fig.suptitle(f"Cover with {thresh} [GBpS] Threshold Over Node Degrees")
    plt.show()