import torch 
from SelfPlay import playrandom
from Imports import device
import numpy as np
import xgboost as xgb
import pandas as pd 
import os, glob

N = 1000
t_dks = torch.load(f"decks_10000.pt")

wins = {'RR':[],'MR':[], 'RM':[], 'MM':[]}
scores = {'RR':[],'MR':[], 'RM':[], 'MM':[]}
df = pd.read_csv("../PRQ_Sizes_Model.csv")
# for i_dck in range(11):

    
for i_dck in range(3):
    
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
                wins[code].append(win)
                scores[code].append(score)
                print(f'{code}{i_dck:<3} | Alex wins = {100*(t_fsc >= 0).sum().item() / N:6.2f}% | Average Score = {t_fsc.sum().item() / N:7.2f}')

print(100*"**")
num = len(wins["RR"])
print(f"Number of Decks: {num}")
for key in wins.keys():
    print(f'{key}: Average Win Rate = {np.array(wins[key]).mean():6.2f} | Average Score = {np.array(scores[key]).mean():7.2f}')

import pdb; pdb.set_trace()