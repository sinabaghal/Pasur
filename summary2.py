import torch 
from SelfPlay import playrandom
from Imports import device
import numpy as np
import xgboost as xgb
import pandas as pd 
import os, glob

N = 5000
t_dks = torch.load(f"decks_10000.pt")

wins   = {'MM':{}, 'MR':{},'RM':{},'RR':{}}
scores = {'MM':{}, 'MR':{},'RM':{},'RR':{}}
df = pd.read_csv("../PRQ_Sizes_Model.csv")
# for i_dck in range(11):
deck_ids = range(100)
    
for i_dck in deck_ids:
    
    # md = df[df.Deck == f"D{i_dck}"].Model.values[0]
    # if md == 'Y':
    print(f'DECK {i_dck}')
    bst = xgb.Booster()
    # folder = f"../MDL/D{i_dck}/model_{i_dck}"
    # pattern = os.path.join(folder, "*_cc.xgb")
    # matches = glob.glob(pattern)

    # model_path = matches[0] 
    bst.load_model(f"../MDL/D{i_dck}/model_{i_dck}.xgb")
    
    t_dck = t_dks[i_dck,:].to(device)
    models = [('R','random'), ('M',bst)]
    # models = [('M',bst)]

    for mdl0 in models:
        for mdl1 in models:

            code = f'{mdl0[0]}{mdl1[0]}'
            t_fsc, t_scr_, t_ltx_ = playrandom(t_dck, N=N, x_alx = mdl0[1] , x_bob = mdl1[1], to_latex = False)
            win   = 100*(t_fsc >= 0).sum().item()/N
            score = t_fsc.sum().item() / N 
            wins[code][i_dck] = win 
            scores[code][i_dck] = score 

data = []           
for deck_id in deck_ids:
    row = {'Deck': deck_id}
    for strat in wins:
        row[f'{strat}_WINS'] = wins[strat].get(deck_id, None)
        row[f'{strat}_SCORES'] = scores[strat].get(deck_id, None)
    data.append(row)
df = pd.DataFrame(data)
import pdb; pdb.set_trace()
# df = pd.DataFrame({'Deck': list(wins.keys()),'Win': list(wins.values()),'Score': [scores[k] for k in wins]}).to_csv("MM.csv")