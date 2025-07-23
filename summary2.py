import torch 
from SelfPlay import playrandom
from Imports import device
import numpy as np
import xgboost as xgb
import pandas as pd 
import os, glob

N = 1000
t_dks = torch.load(f"decks_10000.pt")

wins   = {}
scores = {}
df = pd.read_csv("../PRQ_Sizes_Model.csv")
# for i_dck in range(11):

    
for i_dck in range(152):
    
    md = df[df.Deck == f"D{i_dck}"].Model.values[0]
    if md == 'Y':

        bst = xgb.Booster()
        folder = f"../MDL/D{i_dck}/"
        pattern = os.path.join(folder, "*_cc.xgb")
        matches = glob.glob(pattern)

        model_path = matches[0] 
        bst.load_model(model_path)
        
        t_dck = t_dks[i_dck,:].to(device)
        # models = [('R','random'), ('M',bst)]
        models = [('M',bst)]

        for mdl0 in models:
            for mdl1 in models:

                code = f'{mdl0[0]}{mdl1[0]}'
                t_fsc, t_scr_, t_ltx_ = playrandom(t_dck, N=N, x_alx = mdl0[1] , x_bob = mdl1[1], to_latex = False)
                win   = 100*(t_fsc >= 0).sum().item()/N
                score = t_fsc.sum().item() / N 
                wins[i_dck] = win 
                scores[i_dck] = score 

df = pd.DataFrame({'Deck': list(wins.keys()),'Win': list(wins.values()),'Score': [scores[k] for k in wins]}).to_csv("MM.csv")
import pdb; pdb.set_trace()