import torch, os
from zstd_store import load_tensor
from collections import defaultdict
import pandas as pd 
import matplotlib.pyplot as plt
from Imports import device, INT8, INT32, d_snw, d_scr
import torch.nn.functional as F
import numpy as np 

INF_FDLR = "../DATA2/INF/"
SGM_FDLR = "../DATA2/SGM/"
SID_FDLR = "../DATA2/SID/"
SCR_FDLR = "../DATA2/SCR/"
T52_FDLR = "../DATA2/T52/"
MDL_FDLR = "../DATA2/MDL/"

# FIG_FDLR = "../DATA/FIG/"

def get_csv():

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

        else:
            d_sdh[tu_scr[:2]].append(dh[torch.where(i<t_cum)[0][0].item()])

    d_sdh = {key:list(set(d_sdh[key])) for key in d_sdh.keys()}
    df = pd.DataFrame(columns = ['Alex', 'Bob', 'Win', 'Hand', 'Deck'])
    
    for key, k_dh in d_sdh.items():
        for i_dck, hand in k_dh:
                if isinstance(key,int):
                    
                    df.loc[len(df)] = [0,0,key,hand,i_dck]

                else:
                    df.loc[len(df)] = [key[0],key[1],0,hand,i_dck]
                    
            
        

    df = df.sort_values(by=['Deck','Hand', 'Win', 'Alex', 'Bob']).reset_index(drop=True)
    df.to_csv("scores.csv")


# get_csv()
df    = pd.read_csv("scores.csv")
t_dks = torch.load(f"decks_10000.pt")

def get_data(i_hnd, i_trn, i_ply, tu_scr):

    df2 = df[df.Hand==i_hnd]

    for item in df2.groupby(['Alex', 'Bob', 'Win'])['Deck']:

        t_scr  = torch.tensor(item[0])
        if not torch.all(t_scr == torch.tensor(tu_scr)): continue 
        i_dcks = item[1].values

        l_mdl = []
        l_sgm = []

        for i_dck in i_dcks:

            t_dck = t_dks[i_dck]
            
            t_scs = load_tensor(SCR_FDLR+f"D{i_dck}_scr_{i_hnd}.pt.zst")
            t_m52 = load_tensor(T52_FDLR+f"D{i_dck}_t52_{i_hnd}.pt.zst")
            t_sid = load_tensor(SID_FDLR+f"D{i_dck}_sid_{i_hnd}_{i_trn}_{i_ply}.pt.zst") 
            t_mdl = load_tensor(MDL_FDLR+f"D{i_dck}_mdl_{i_hnd}_{i_trn}_{i_ply}.pt.zst")
            t_sgm = load_tensor(SGM_FDLR+f"D{i_dck}_sgm_{i_hnd}_{i_trn}_{i_ply}.pt.zst")

            t_msk = (t_scs[:,[0,1,3]] == t_scr).all(dim=1)
            t_nnz = t_msk.nonzero().flatten()
            t_smk = torch.isin(t_sid[:, 1], t_nnz)

            t_ssi = t_sid[t_smk]
            
            t_sgm = t_sgm[t_ssi[:,0]]
            t_mdl = t_mdl[t_ssi[:,0]]

            t_nnd = torch.zeros((1,52))
            if i_hnd > 0:
                
                i_nnd = 8*i_hnd+4
                t_nnd[0,t_dck[:i_nnd]] = 1

            t_tmp   = torch.zeros((t_ssi.shape[0],52), dtype=torch.int8)
            t_tmp[:,t_m52] = t_mdl
            t_tmp[torch.logical_and(t_nnd==1, t_tmp==0)] = -127

            t_mdl = t_tmp

            l_mdl.append(t_mdl) 
            l_sgm.append(t_sgm) 
            
            
            
        

        return l_mdl, l_sgm
        # t_mdl, t_inv = torch.unique(t_mdl, return_inverse = True, dim=0)
        # import pdb; pdb.set_trace()

if __name__ == "__main__":

    i_hnd, i_trn, i_ply = 5, 3, 0
    l_mdl, l_sgm = get_data(i_hnd, i_trn, i_ply, (0,0,1))
    t_mdl = torch.cat(l_mdl,dim=0).numpy()
    data = {f"C{i}": t_mdl[:, i] for i in range(t_mdl.shape[1])}
    df = pd.DataFrame(data)
    df.to_parquet("tensor_int8.parquet", engine="pyarrow")
    merged_df.to_parquet(
    "partitioned_data/",
    engine="pyarrow",
    partition_on=group_cols,
    write_index=False)

    # for limit in range(10, 200,10):





        # t_mdl = torch.cat(l_mdl[:limit],dim=0)
        # i_cnt = t_mdl.unique(dim=0).shape[0]
        # print(limit, i_cnt, i_cnt/t_mdl.shape[0])
        # import pdb; pdb.set_trace()
        # t_sgm = torch.cat(l_sgm[:b],dim=0)

    import pdb; pdb.set_trace()
# df.to_csv("scores.csv")
# df_g = df.groupby(['Alex', 'Bob', 'Win', 'Hand'])['Deck'].nunique().reset_index()
# df_g.rename(columns={'Deck': 'DistinctDecks'}, inplace=True)
# df_g = df_g.sort_values(by=['DistinctDecks'], ascending=False).reset_index(drop=True)

# with pd.ExcelWriter("scores.xlsx") as writer:

#     df.to_excel(writer, sheet_name='Main', index=False)
#     df_g.to_excel(writer, sheet_name='Deck Count', index=False)
# import pdb; pdb.set_trace()

# df_g['Label'] = df_g[['Alex', 'Bob', 'Win', 'Hand']].astype(str).agg('-'.join, axis=1)

# plt.figure(figsize=(14, 6))
# plt.bar(df_g['Label'], df_g['DistinctDecks'])
# plt.xticks(rotation=90)
# # plt.ylabel("Number of Distinct Decks")
# # plt.title("Distinct Deck Counts per (Alex, Bob, Win, Hand) Combination")
# plt.tight_layout()
# plt.show()

