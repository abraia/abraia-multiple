
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = [12, 8]
# plt.rcParams['figure.dpi'] = 100

from dotenv import load_dotenv
load_dotenv()

from .multiple import Multiple
from . import data
from . import hsi

__all__ = ['np', 'pd', 'hsi', 'data', 'Multiple']
