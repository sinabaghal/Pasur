import torch, sys, os
import torch.nn.functional as F
from Utils import get_d_mask, pad_helper
from FindMoves import find_moves
from ApplyMoves import apply_moves
from Imports import device, INT8, INT32, d_snw, d_scr
from RepeatedBlocks import repeatedblocks
from CleanPool import cleanpool 
from tqdm import trange 
import dask.dataframe as dd
import shutil
# import dask
# from dask.distributed import Client
from dask import delayed, compute
# from joblib import Parallel, delayed
# import multiprocessing

# num_workers = multiprocessing.cpu_count()

# from Setup import setup
# from ExternalSampling import extsampling, sorttsid
from zstd_store import to_memory
import pandas as pd 
import time 
import numpy as np
# import xgboost as xgb
# import numpy as np
# import matplotlib.pyplot as plt
# from zstd_store import load_tensor 
# from SelfPlay import playrandom
# from GameToLatex import tp



    
def get_col(i):

    if i<52:
        return f"C{i}"
    elif i == 52: return "SGM"
    elif i == 53: return "A"
    elif i == 54: return "B"
    elif i == 55: return "W"
    elif i == 56: return "H"
    elif i == 57: return "T"
    elif i == 58: return "P"
    elif i == 59: return "D"



def get_prq(t_mdl, t_sid, t_scr, t_m52, i_hnd,i_trn,i_ply):

    
    t_tmp = torch.cat([t_m52,torch.tensor([False, False,False, False,False,False,False,False])])
    # t_scr = t_scr[:,[0,1,3]].clone()

    t_prq = torch.zeros((t_sid.shape[0], 60), dtype=torch.int8)
    t_prq[:,t_tmp]       = t_mdl[t_sid[:,0]]
    # t_prq[:,52]          = t_prq[:,52].astype("int32")
    t_prq[:,53:56]       = t_scr[t_sid[:,1]][:,[0,1,3]] #53,54,55
    t_prq[:,56:59]       = torch.tensor([i_hnd,i_trn,i_ply], dtype=torch.int8) #56,57,58
    t_prq                = t_prq.cpu()
    data = {get_col(i): t_prq[:, i] for i in range(t_prq.shape[1])}
    df   = pd.DataFrame(data)
    df["D"] = df["D"].astype("int32")
    df["D"] = np.full(len(df), i_dck, dtype=np.int32)
    # ddf = dd.from_pandas(pd.DataFrame(data), npartitions=1)
    return df


i_cods = [
            f'{i_hnd}_{i_trn}_{i_ply}'
            for i_hnd in range(6)
            for i_trn in range(4)
            for i_ply in range(2)
            ]

i_hnds = range(6)

os.makedirs(f"../DATA/MDL/", exist_ok=True)
# os.makedirs(f"../DATA/INF/", exist_ok=True)
os.makedirs(f"../DATA/SGM/", exist_ok=True)
os.makedirs(f"../DATA/SID/", exist_ok=True)
os.makedirs(f"../DATA/SCR/", exist_ok=True)
os.makedirs(f"../DATA/FIG/", exist_ok=True)
os.makedirs(f"../DATA/T52/", exist_ok=True)
time.sleep(5)

def simulategamerun(t_stk, i_dck):

    
   
 
    d_fls = {}
    t_scrs, t_mdls, t_sids, t_m52s = {}, {}, {}, {}
    # t_scrs, t_infs, t_sids, t_m52s = {}, {}, {}, {}
    # t_nns   = {}

    t_m52 = torch.tensor([True if i in t_stk[:4] else False for i in range(52)], device=device)
    # t_mdd = torch.tensor([False for _ in range(52)], device=device)
    
    t_inf        = torch.zeros((1,3,4), dtype=INT8, device=device)
    t_inf[0,0,:] = 3
    
    t_dck   = t_stk.clone()
    t_stk   = t_stk[4:]

    t_scr   = torch.zeros((1,len(d_scr)), device=device,dtype=INT8)
    t_sid   = torch.zeros((1,2), device=device,dtype=torch.int32)
    # dfs    = {}
    for i_hnd in i_hnds:
        torch.cuda.empty_cache()

        t_nnd = torch.zeros((1,52), device=device)

        t_scrs[i_hnd] = t_scr.cpu()
        if i_hnd > 0:
            
            i_nnd = 8*i_hnd+4
            t_nnd[0,t_dck[:i_nnd]] = 1


        torch.cuda.empty_cache()
        _, c_scr = torch.unique(t_sid[:,0],dim=0,return_counts=True)

        d_msk = get_d_mask(t_m52, t_stk[:8])
        t_stk = t_stk[8:]
        t_inf = F.pad(t_inf, (0, 8))[:,:,d_msk['pad']]
        t_inf[:,0, d_msk['nxt_a']] = 1
        t_inf[:,0, d_msk['nxt_b']] = 2
        i_pad_n, _       = pad_helper(d_msk, 'n') 
        i_pad_k, t_pad_k = pad_helper(d_msk, 'k')
        i_pad_q, t_pad_q = pad_helper(d_msk, 'q')
        i_pad_j, _       = pad_helper(d_msk, 'j')

        d_pad = {'i_pad_n': i_pad_n, 'i_pad_k':i_pad_k, 'i_pad_q':i_pad_q, 'i_pad_j':i_pad_j, 't_pad_k':t_pad_k, 't_pad_q':t_pad_q}
        t_snw = torch.zeros((t_inf.shape[0],len(d_snw)), device=device,dtype=INT8)
        # print(t_m52)
        t_m52s[i_hnd] = t_m52.clone()
        # if i_hnd == 5:
        #     import pdb; pdb.set_trace()
        for i_trn in range(4):
            for i_ply in range(2):


                i_cod         = f'{i_hnd}_{i_trn}_{i_ply}'
                i_sid = t_sid.shape[0]
                t_act, c_act  = find_moves(t_inf,i_ply, d_msk, d_pad) 

                t_inf         = torch.repeat_interleave(t_inf, c_act, dim=0)
                t_snw         = torch.repeat_interleave(t_snw, c_act, dim=0)

                t_inf, t_snw  = apply_moves(t_inf, t_act, t_snw, d_msk,  i_hnd, i_ply, i_trn)
                
                t_cl1         = repeatedblocks(t_sid[:,1], c_scr, c_act)
                t_edg         = repeatedblocks(torch.arange(t_sid.shape[0], device=device),c_scr,c_act)
                c_edg         = c_act.repeat_interleave(c_scr)
                c_scr         = torch.repeat_interleave(c_scr, c_act, dim=0)
                t_cl0         = torch.repeat_interleave(torch.arange(t_inf.shape[0],device=device), c_scr)
                t_sid         = torch.stack([t_cl0,t_cl1], dim=1)


                ## CONSTRUCT INFOSET I 
                
                t_mdl         = t_inf[:,1,:]-t_inf[:,2,:]
                t_mdl[torch.logical_and(t_inf[:,0,:]==3, t_mdl==0)] = 110 ### Card was already in the pool 
                t_mdl[torch.logical_and(t_inf[:,0,:]==i_ply+1, t_mdl==0)] = 100  ### Player has the card


                t_sgm = 1/torch.repeat_interleave(c_edg, c_edg)

                # assert t_sgm.shape[0] == t_mdl.shape[0]

                d_fls[f'i_sid_{i_cod}'] = i_sid #TODO
                d_fls[f't_sgm_{i_cod}'] = t_sgm.contiguous().to(torch.float32) #TODO
                d_fls[f't_edg_{i_cod}'] = t_edg.contiguous() #TODO
                # d_fls[f't_sgm_{i_cod}'] = torch.round(t_sgm*255).to(torch.uint8)
                # d_fls[f'c_edg_{i_cod}'] = c_edg
                # d_fls[f'c_act_{i_cod}'] = c_act
                
                t_mdls[i_cod]  = t_mdl.clone()
                t_sids[i_cod]  = t_sid.clone()


                # t_nns[i_cod]  = t_mdl
                # t_infs[i_cod]  = t_inf
        

        # d_scr = {'a_clb':0, 'b_clb':1, 'pts_dlt':2, 'max_clb':3}
        # d_snw = {'a_clb':0,'b_clb':1,'pts_dlt':2, 'lst_pck':3,'a_pts':4,'a_sur':5,'b_pts':6,'b_sur':7}
        
        if i_hnd ==  5:  t_inf, t_snw = cleanpool(t_inf, t_snw, d_msk)

        t_inf        = t_inf[:,0,:]
        t_snw[:,d_snw['pts_dlt']]   = t_snw[:,d_snw['a_pts']] + t_snw[:,d_snw['a_sur']] - t_snw[:,d_snw['b_pts']] - t_snw[:,d_snw['b_sur']]
        d_fls[f't_snw_{i_hnd}']     = t_snw[:,2].contiguous()
        
        t_snw[:,2]   = 0 
        t_snw        = t_snw[:,:4]
        t_snw[:,-1]  = 0    


        t_inf, t_lnk = torch.unique(t_inf,dim=0, sorted=False, return_inverse=True)
        d_fls[f't_lnf_{i_hnd}'] = t_lnk.contiguous()
        
        t_snw, t_wid = torch.unique(t_snw,dim=0, sorted=False, return_inverse=True) 
        t_prs        = torch.stack([t_sid[:,1], t_wid[t_sid[:,0]]],dim=1)
        t_sid[:,0]   = t_lnk[t_sid[:,0]]

        t_prs,t_pid  = torch.unique(t_prs,dim=0,sorted=False,return_inverse=True)
        t_scr        = t_scr[t_prs[:,0]]+t_snw[t_prs[:,1]]

        t_mal = t_scr[:,d_scr['a_clb']]>=7
        t_mbb = t_scr[:,d_scr['b_clb']]>=7
        t_scr[:,d_scr['max_clb']][t_mal]= 1
        t_scr[:,d_scr['max_clb']][t_mbb]= 2
        t_mcl = t_scr[:,d_scr['max_clb']]>0
        t_scr[:,0:2].masked_fill_(t_mcl.unsqueeze(-1), 0)


        t_scr,t_fid  = torch.unique(t_scr, dim=0,sorted=False,return_inverse=True)
        t_sid[:,1]   = t_fid[t_pid]
        t_sid, t_lnk = torch.unique(t_sid,dim=0,return_inverse=True)
        # if saving:  to_zstd([t_lnk], ['t_lnk'], folder, i_hnd) 
        d_fls[f't_lnk_{i_hnd}'] = t_lnk
        
        t_c52        = t_m52.clone()
        t_m52[t_c52] = torch.any(t_inf>0, dim=0)
        t_inf        = t_inf[:,t_m52[t_c52]]
        # t_ndd        = torch.logical_and(t_c52,~t_m52)
        # t_mdd[t_ndd] = True
        t_inf        = t_inf.unsqueeze(1)
        
        if t_inf.shape[2] == 0:
            t_inf = torch.empty((t_inf.shape[0], 3, 0), device='cuda:0', dtype=INT8)
        else:
            t_inf = F.pad(t_inf, (0, 0, 0, 2))

    d_fls['t_scr_6'], d_fls['t_sid_6'] = t_scr.contiguous(), t_sid.contiguous() #TODO

    
    
    [to_memory(t_mdls[i_cod].cpu(), f"../DATA/MDL/D{i_dck}_mdl_{i_cod}.pt.zst") for i_cod in i_cods]
    [to_memory(t_sids[i_cod].cpu(), f"../DATA/SID/D{i_dck}_sid_{i_cod}.pt.zst") for i_cod in i_cods]
    [to_memory(t_scrs[i_hnd].cpu(), f"../DATA/SCR/D{i_dck}_scr_{i_hnd}.pt.zst") for i_hnd in i_hnds]
    [to_memory(t_m52s[i_hnd].cpu(), f"../DATA/T52/D{i_dck}_t52_{i_hnd}.pt.zst") for i_hnd in i_hnds]
    
    return d_fls

if __name__ == "__main__":

    
    

    # @profile
    def find_cfr(i_dck, d_fls):

        
       
       
        all_sgm = torch.cat([d_fls[f't_sgm_{id}'] for id in i_cods], dim=0)
        t_scr, t_sid = d_fls['t_scr_6'], d_fls['t_sid_6']
        t_fsc = 7*(2*(t_scr[:,-1] % 2)-1)
        
        # t_fsc = 7*(2*(t_scr[:,-1] % 2)-1)+t_scr[:,-2]
        t_vll = t_fsc[t_sid[:,1]].to(torch.float32)

        cnt = 0
        maxes, means = [], []
        # a_sgm = []

        pbar = trange(1000, desc=f"Processing Deck {i_dck}")
        for cnt in pbar:

            t_val = t_vll

            for i_hnd in reversed(range(6)):

                t_lnk  = d_fls[f't_lnk_{i_hnd}'] 
                t_snw  = d_fls[f't_snw_{i_hnd}']
                t_val  = t_val[t_lnk] + t_snw[t_lnk]


                for i_trn in reversed(range(4)):
                    for i_ply in reversed(range(2)):

                        torch.cuda.empty_cache()

                        i_cod               = f'{i_hnd}_{i_trn}_{i_ply}'
                        i_sid    = d_fls[f'i_sid_{i_cod}'] 
                        
                        t_sum        = torch.zeros(i_sid, dtype = torch.float32, device=device)  
                        t_edg, t_sgm  = d_fls[f't_edg_{i_cod}'],  d_fls[f't_sgm_{i_cod}']


                        t_sum.scatter_add_(0, t_edg, t_sgm*t_val)
                        
                        t_reg         = t_val - t_sum[t_edg]
                        t_val         = t_sum
                        t_reg         = torch.clamp(t_reg, min=0.01)
                        
                    
                        t_sum         = torch.zeros(i_sid, dtype = torch.float32, device=device)  
                        t_sum.scatter_add_(0, t_edg, t_reg)
                        t_sgm         = t_reg/t_sum[t_edg]

                        d_fls[f't_sgm_{i_cod}'] = t_sgm

                        del t_reg
                        
            
            next_all_sgm = torch.cat([d_fls[f't_sgm_{id}'] for id in i_cods], dim=0)
            all_diff = torch.abs(all_sgm - next_all_sgm)

            new_mean = torch.mean(all_diff)
            new_maxx = torch.max(all_diff)

            all_sgm = next_all_sgm
            means.append(new_mean); maxes.append(new_maxx)

            if cnt % 10 ==0 : 

                pbar.set_postfix(max=new_maxx.item())
                if new_maxx < 0.01:
                    [to_memory(d_fls[f't_sgm_{id}'].cpu(), f"../DATA/SGM/D{i_dck}_sgm_{id}.pt.zst") for id in i_cods]
                    df = pd.DataFrame({"mean": torch.tensor(means).cpu().numpy(),"max": torch.tensor(maxes).cpu().numpy()})
                    df.to_csv(f"../DATA/FIG/D{i_dck}_mean_max.csv", index=False)
                    return 
                   
            cnt +=  1
                 

        [to_memory(d_fls[f't_sgm_{i_cod}'].cpu(), f"../DATA/SGM/D{i_dck}_sgm_{i_cod}.pt.zst") for i_cod in i_cods]
        df = pd.DataFrame({"mean": torch.tensor(means).cpu().numpy(),"max": torch.tensor(maxes).cpu().numpy()})
        df.to_csv(f"../DATA/FIG/D{i_dck}_mean_max.csv", index=False)



    N = 50
    t_dks = torch.load(f"decks_10000.pt")
    for i_dck in range(N): 

        torch.cuda.empty_cache()
        # try:

        t_stk = t_dks[i_dck,:].to(device)
        d_fls  = simulategamerun(t_stk, i_dck)
        find_cfr(i_dck, d_fls)
        