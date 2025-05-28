import torch, sys, os, math
import torch.nn.functional as F
from FindMoves import find_moves
from ApplyMoves import apply_moves
from Imports import device, INT8, INT32, d_snw, d_scr
from CleanPool import cleanpool 
from NeuralNet import SNN, init_weights_zero
import torch.nn as nn
from Utils import pad_helper
from tqdm import trange
import xgboost as xgb
from tqdm import trange, tqdm
import numpy as np 
from zstd_store import load_tensor 
import matplotlib.pyplot as plt




if __name__ == "__main__":
  
    t0_win, _ = selfplay(N=1000, to_latex=True, d_fls = {})
    
