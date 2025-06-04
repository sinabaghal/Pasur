import torch, os
from zstd_store import load_tensor
from collections import defaultdict
import pandas as pd 
import matplotlib.pyplot as plt


INF_FDLR = "../DATA/INF/"
SGM_FDLR = "../DATA/SGM/"
SID_FDLR = "../DATA/SID/"
SCR_FDLR = "../DATA/SCR/"
FIG_FDLR = "../DATA/FIG/"
d_sdh    = defaultdict(list)

f_scrs  = os.listdir(SCR_FDLR)
t_scrs  = [load_tensor(SCR_FDLR+file) for file in f_scrs]
i_shps  = torch.tensor([t.shape[0] for t in t_scrs])
t_cum   = torch.cumsum(i_shps,dim=0)
dh      = [(int(file.split('_')[0][1:]), int(file.split('.')[0][-1])) for file in f_scrs] # DECK, HAND
tc_scrs = torch.cat(t_scrs,dim=0)

n_scrs, t_inv = torch.unique(tc_scrs, sorted = False, return_inverse = True, dim = 0)
for i in range(t_inv.shape[0]):

    tu_scr = tuple(n_scrs[t_inv[i]].tolist())
    if tu_scr[-1] >0:

        d_sdh[tu_scr[-1]].append(dh[torch.where(i<t_cum)[0][0].item()])

        # i_scr = tu_scr[-2]
        
        # if tu_scr[-1] == 1:
        
        #     i_scr += 7 
        # else:
        #     i_scr -= 7

        # i_scr = min(max(i_scr,-20),20)

        # n_dh = dh[torch.where(i<t_cum)[0][0].item()]

        # # if n_dh[-1] == 5:
        # #     i_scr = min(max(i_scr,-13),13)
        # # else:
        # #     i_scr = min(max(i_scr,-20),20)

        # d_sdh[i_scr].append(n_dh)

    else:
        d_sdh[tu_scr[:2]].append(dh[torch.where(i<t_cum)[0][0].item()])
        # d_sdh[tu_scr].append(dh[torch.where(i<t_cum)[0][0].item()])

d_sdh = {key:list(set(d_sdh[key])) for key in d_sdh.keys()}

df = pd.DataFrame(columns = ['Alex', 'Bob', 'Win', 'Hand', 'Deck'])

# import pdb; pdb.set_trace()
for key, k_dh in d_sdh.items():
   
    for deck, hand in k_dh:
            
            if isinstance(key,int):
                 
                df.loc[len(df)] = [0,0,key,hand,deck]

            else:
                 
                df.loc[len(df)] = [key[0],key[1],0,hand,deck]
                  
         
    

df = df.sort_values(by=['Deck','Hand', 'Win', 'Alex', 'Bob']).reset_index(drop=True)
# df.to_csv("scores.csv")
df_g = df.groupby(['Alex', 'Bob', 'Win', 'Hand'])['Deck'].nunique().reset_index()
df_g.rename(columns={'Deck': 'DistinctDecks'}, inplace=True)
df_g = df_g.sort_values(by=['DistinctDecks'], ascending=False).reset_index(drop=True)

with pd.ExcelWriter("scores.xlsx") as writer:

    df.to_excel(writer, sheet_name='Main', index=False)
    df_g.to_excel(writer, sheet_name='Deck Count', index=False)
# import pdb; pdb.set_trace()

# df_g['Label'] = df_g[['Alex', 'Bob', 'Win', 'Hand']].astype(str).agg('-'.join, axis=1)

# plt.figure(figsize=(14, 6))
# plt.bar(df_g['Label'], df_g['DistinctDecks'])
# plt.xticks(rotation=90)
# # plt.ylabel("Number of Distinct Decks")
# # plt.title("Distinct Deck Counts per (Alex, Bob, Win, Hand) Combination")
# plt.tight_layout()
# plt.show()

