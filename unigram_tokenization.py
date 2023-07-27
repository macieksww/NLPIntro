import pandas as pd
import copy
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from helpful_functions import read_file
from collections import Counter
pd.options.mode.chained_assignment = None

class UnigramTokenizer:
    def __init__(self, path_to_txt="txt/sample.txt"):
        self.text = read_file(path_to_txt ,"r")