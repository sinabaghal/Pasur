import torch, sys, os
import torch.nn.functional as F
from Utils import get_d_mask, pad_helper
from FindMoves import find_moves
from ApplyMoves import apply_moves
from Imports import device, INT8, INT32, d_rus, d_scr
from RepeatedBlocks import repeatedblocks
from CleanPool import cleanpool 
import math
from tqdm import trange 
import dask.dataframe as dd
import pandas as pd 
import time 
import numpy as np
import matplotlib.pyplot as plt;
import gc 
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

def simulategamerun(t_stk):

    t_dck   = t_stk.clone()
    d_fls = {}
    t_scrs, t_cmps, t_fgms, t_m52s = {}, {}, {}, {}
    # t_scrs, t_infs, t_fgms, t_m52s = {}, {}, {}, {}
    # t_nns   = {}

    t_m52 = torch.tensor([True if i in t_stk[:4] else False for i in range(52)], device=device)
    # t_mdd = torch.tensor([False for _ in range(52)], device=device)
    
    t_gme        = torch.zeros((1,3,4), dtype=INT8, device=device)
    t_gme[0,0,:] = 3
    
    t_stk   = t_stk[4:]

    t_scr   = torch.zeros((1,len(d_scr)), device=device,dtype=INT8)
    t_fgm   = torch.zeros((1,2), device=device,dtype=torch.int32)
    # i_sid = t_fgm.shape[0]
    # dfs    = {}
    for i_hnd in i_hnds:
        torch.cuda.empty_cache()

        t_dlt = torch.zeros((1,52), device=device)

        t_scrs[i_hnd] = t_scr.to(device='cpu')
        if i_hnd > 0:
            
            i_nnd = 8*i_hnd+4
            t_dlt[0,t_dck[:i_nnd]] = 1


        torch.cuda.empty_cache()
        _, c_scr = torch.unique(t_fgm[:,0],dim=0,return_counts=True)

        d_msk = get_d_mask(t_m52, t_stk[:8])
        t_stk = t_stk[8:]
        t_gme = F.pad(t_gme, (0, 8))[:,:,d_msk['pad']]
        t_gme[:,0, d_msk['nxt_a']] = 1
        t_gme[:,0, d_msk['nxt_b']] = 2
        # print(i_hnd, t_gme.shape[0])
        i_pdn, _     = pad_helper(d_msk, 'n') 
        i_pdk, t_pdk = pad_helper(d_msk, 'k')
        i_pdq, t_pdq = pad_helper(d_msk, 'q')
        i_pdj, _     = pad_helper(d_msk, 'j')

        d_pad = {'i_pdn': i_pdn, 'i_pdk':i_pdk, 'i_pdq':i_pdq, 'i_pdj':i_pdj, 't_pdk':t_pdk, 't_pdq':t_pdq}
        t_rus = torch.zeros((t_gme.shape[0],len(d_rus)), device=device, dtype=INT8)
        # print(t_m52)
        t_m52s[i_hnd] = torch.cat([t_m52.clone(),torch.tensor([False, False,False, False,False,False,False,False], device=device)])
        # if i_hnd == 5:
        #     import pdb; pdb.set_trace()
        for i_trn in range(4):
            for i_ply in range(2):

                gc.collect()
                torch.cuda.empty_cache()
                i_cod         = f'{i_hnd}_{i_trn}_{i_ply}'
                # print(i_cod)
                i_sid = t_fgm.shape[0]
                t_act, t_brf  = find_moves(t_gme,i_ply, d_msk, d_pad) 
                

                t_gme         = torch.repeat_interleave(t_gme, t_brf, dim=0)
                t_rus         = torch.repeat_interleave(t_rus, t_brf, dim=0)

                t_gme, t_rus  = apply_moves(t_gme, t_act, t_rus, d_msk,  i_hnd, i_ply, i_trn)
                
                
                t_edg         = repeatedblocks(torch.arange(t_fgm.shape[0], device=device),c_scr,t_brf)
                c_edg         = t_brf.repeat_interleave(c_scr)
                t_sgm = 1/torch.repeat_interleave(c_edg, c_edg)
                d_fls[f'c_edg_{i_cod}'] = c_edg.to(device='cpu')
                del c_edg

                t_cl1         = repeatedblocks(t_fgm[:,1], c_scr, t_brf)
                c_scr         = torch.repeat_interleave(c_scr, t_brf, dim=0)
                del t_brf


                t_cl0         = torch.repeat_interleave(torch.arange(t_gme.shape[0],device=device), c_scr)
                t_fgm         = torch.stack([t_cl0,t_cl1], dim=1)
                # i_sid = t_fgm.shape[0]

                del t_cl0 , t_cl1
                ## CONSTRUCT INFOSET I 
                
                t_cmp         = t_gme[:,1,:]-t_gme[:,2,:]
                # t_lpm         = torch.logical_and(t_gme[:,0,:]==0, t_cmp!= 0)
                t_lpm         = torch.logical_and(t_gme[:, 0, :] == 0, (t_cmp != 0) & (t_cmp.abs() < 5))
                t_cmp[t_lpm]  =  110+t_cmp[t_lpm] # Lay Pick Mask
                t_cmp[torch.logical_and(t_gme[:,0,:]==3, t_cmp==0)] = 110 ### Card was already in the pool 
                t_cmp[torch.logical_and(t_gme[:,0,:]==1, t_cmp==0)] = 100  ### Player has the card
                t_cmp[torch.logical_and(t_gme[:,0,:]==2, t_cmp==0)] = 105  ### Player has the card

                # assert t_sgm.shape[0] == t_cmp.shape[0]

                d_fls[f'i_sid_{i_cod}'] = i_sid #TODO
                # d_fls[f't_sgm_{i_cod}'] = torch.zeros_like(t_sgm, device=device).to(torch.float32)
                d_fls[f't_sgm_{i_cod}'] = t_sgm.to(torch.float32).to(device='cpu') #TODO:64
                # d_fls[f't_sgm_{i_cod}'] = t_sgm.to(torch.float32).to(device='cpu') #TODO:64
                d_fls[f't_edg_{i_cod}'] = t_edg.contiguous().to(device='cpu') #TODO
                # d_fls[f't_edg_{i_cod}'] = t_edg.contiguous().to(device='cpu') #TODO
                # d_fls[f't_sgm_{i_cod}'] = torch.round(t_sgm*255).to(torch.uint8)
                # d_fls[f'c_edg_{i_cod}'] = c_edg.contiguous().to(device='cpu')
                # d_fls[f't_brf_{i_cod}'] = t_brf
                
                t_cmps[i_cod]  = t_cmp.to(device='cpu')
                t_fgms[i_cod]  = t_fgm.to(device='cpu')
                del t_cmp, t_sgm, t_lpm
                # t_cmps[i_cod]  = t_cmp.clone().to(device='cpu')
                # t_fgms[i_cod]  = t_fgm.clone().to(device='cpu')
                # d_fls[f'i_sid_{i_cod}'] = t_fgm.shape[0] #TODO


                # t_nns[i_cod]  = t_cmp
                # t_infs[i_cod]  = t_gme
        

        # d_scr = {'a_clb':0, 'b_clb':1, 'pts_dlt':2, 'max_clb':3}
        # d_rus = {'a_clb':0,'b_clb':1,'pts_dlt':2, 'lst_pck':3,'a_pts':4,'a_sur':5,'b_pts':6,'b_sur':7}
        # print(t_gme.shape[0])
        # print(t_fgm.shape[0])
        # print(100*"**")
        # print('here')
        
        gc.collect();torch.cuda.empty_cache()
        
        if i_hnd ==  5:  
            
            # pass
            t_gme, t_rus = cleanpool(t_gme, t_rus, d_msk)


        t_gme                       = t_gme[:,0,:]
        gc.collect();torch.cuda.empty_cache()

        # t_gme_c = t_gme.clone()
        t_rus[:,d_rus['pts_dlt']]   = t_rus[:,d_rus['a_pts']] + t_rus[:,d_rus['a_sur']] - t_rus[:,d_rus['b_pts']] - t_rus[:,d_rus['b_sur']]
        # if i_hnd ==  5: 
        d_fls[f't_rus_{i_hnd}']     = torch.repeat_interleave(t_rus[:,d_rus['pts_dlt']],c_scr, dim=0).to(device='cpu')
        # d_fls[f't_rus_{i_hnd}']     = torch.repeat_interleave(t_rus[:,d_rus['pts_dlt']],c_scr, dim=0).contiguous().to(device='cpu')
        
        # d_rus = {'a_clb':0,'b_clb':1,'pts_dlt':2, 'lst_pck':3,'a_pts':4,'a_sur':5,'b_pts':6,'b_sur':7}
        # d_scr = {'a_clb':0, 'b_clb':1, 'pts_dlt':2, 'max_clb':3}
        
        t_rus         = t_rus[:,:4]
        t_rus[:,-1]   = 0    
        t_rus[:,-2]   = 0 
        t_rus_, t_wid = torch.unique(t_rus,dim=0, sorted=False, return_inverse=True) 
        
        t_gme, t_lnk = torch.unique(t_gme,dim=0, sorted=False, return_inverse=True)
        # t_rus        = t_rus[t_lnk]
        
        d_fls[f't_lnf_{i_hnd}'] = t_lnk.to(device='cpu')
        # d_fls[f't_lnf_{i_hnd}'] = t_lnk.contiguous().to(device='cpu')

        t_prs        = torch.stack([t_fgm[:,1], t_wid[t_fgm[:,0]]],dim=1)

        t_prs,t_pid  = torch.unique(t_prs,dim=0,sorted=False,return_inverse=True)
        t_scr        = t_scr[t_prs[:,0]]+t_rus_[t_prs[:,1]]

        t_mal = t_scr[:,d_scr['a_clb']]>=7
        t_mbb = t_scr[:,d_scr['b_clb']]>=7

        t_scr[:,d_scr['max_clb']][t_mal]= 1
        t_scr[:,d_scr['max_clb']][t_mbb]= 2
        
        t_mcl = t_scr[:,d_scr['max_clb']]>0
        t_scr[:,0:2].masked_fill_(t_mcl.unsqueeze(-1), 0)

        t_scr, t_fid  = torch.unique(t_scr, dim=0,sorted=False,return_inverse=True)
        t_fgm[:,1]   = t_fid[t_pid] # find pair first and for that pair find score 
        t_fgm[:,0]   = t_lnk[t_fgm[:,0]] # first will become 208 etc 
        
        t_fgm, t_lnk = torch.unique(t_fgm,dim=0,return_inverse=True)
        # print(t_gme.shape[0])
        # print(t_fgm.shape[0])
        # print(100*"**")

        d_fls[f't_lnk_{i_hnd}'] = t_lnk.to(device='cpu')
        d_fls[f't_fgm_{i_hnd}'] = t_fgm.shape[0]
        
        t_c52        = t_m52.clone()
        t_m52[t_c52] = torch.any(t_gme>0, dim=0)
        t_gme        = t_gme[:,t_m52[t_c52]]
        t_gme        = t_gme.unsqueeze(1)
        
        if t_gme.shape[2] == 0:
            t_gme = torch.empty((t_gme.shape[0], 3, 0), device='cuda:0', dtype=INT8)
        else:
            t_gme = F.pad(t_gme, (0, 0, 0, 2))

    d_fls['t_scr_6'], d_fls['t_fgm_6'] = t_scr.contiguous().to(device='cpu'), t_fgm.contiguous().to(device='cpu') #TODO
    
    return d_fls, t_cmps, t_fgms, t_scrs, t_m52s



def find_reach_probs(d_fls, dt_edg, tr_sgm, i_cfr, i_hnd):

    i_h = d_fls[f't_fgm_{i_hnd -1}'] if i_hnd > 0 else 1
    t_rpr = torch.ones(i_h, dtype = torch.float32, device=device) #TODO:64
    d_rpr = {}
    for i_trn in range(4):
        for i_ply in range(2):
            
            i_cod            = f'{i_hnd}_{i_trn}_{i_ply}'
            # import pdb; pdb.set_trace()
            t_sgm = torch.ones_like(tr_sgm[f't_sgm_{i_cod}']) if i_ply == i_cfr else tr_sgm[f't_sgm_{i_cod}']
            t_edg          = dt_edg[f't_edg_{i_cod}']
            # import pdb; pdb.set_trace()
            d_rpr[i_cod]   = t_sgm*t_rpr[t_edg]
            t_rpr          = d_rpr[i_cod].clone()

    return d_rpr

def find_reach_probs2(d_fls, tr_sgm, i_cfr):


    d_rpr = {}
    t_rpr = torch.ones(1, dtype= torch.float32, device=device)
    for i_hnd in range(6):
        for i_trn in range(4):
            for i_ply in range(2):
                
                i_cod            = f'{i_hnd}_{i_trn}_{i_ply}'
                t_sgm = torch.ones_like(d_fls[f't_sgm_{i_cod}']) if i_ply == i_cfr else tr_sgm[f't_sgm_{i_cod}']
                # if i_hnd == 1: import pdb; pdb.set_trace()
                t_edg          = d_fls[f't_edg_{i_cod}']
                d_rpr[i_cod]   = t_sgm*t_rpr[t_edg]
                # print(d_fls[f'i_sid_{i_cod}'], t_rpr.shape[0])
                # assert d_fls[f'i_sid_{i_cod}'] ==  t_rpr.shape[0]
                t_rpr          = d_rpr[i_cod].clone()
    

        t_lnk = d_fls[f't_lnk_{i_hnd}']  
        t_sum = torch.zeros(d_fls[f't_fgm_{i_hnd}'], dtype = torch.float32, device=device)  
        t_sum.scatter_add_(dim=0, index=t_lnk, src=t_rpr)
        t_rpr = t_sum.clone()
        

    return d_rpr



# def find_reach_probs3(d_fls, tr_sgm):


#     d_rpr = {}
#     t_rpr = torch.ones(1, dtype= torch.float64, device=device)
#     for i_hnd in range(6):
#         for i_trn in range(4):
#             for i_ply in range(2):
                
#                 i_cod = f'{i_hnd}_{i_trn}_{i_ply}'
#                 t_sgm =  tr_sgm[f't_sgm_{i_cod}']
#                 t_edg          = d_fls[f't_edg_{i_cod}']
#                 d_rpr[i_cod]   = t_sgm*t_rpr[t_edg]
#                 t_rpr          = d_rpr[i_cod].clone()
#                 # print(t_rpr.sum().item())
    

#         t_lnk = d_fls[f't_lnk_{i_hnd}']  
#         t_sum = torch.zeros(d_fls[f't_fgm_{i_hnd}'], dtype = torch.float64, device=device)  
#         t_sum.scatter_add_(dim=0, index=t_lnk, src=t_rpr)
#         t_rpr = t_sum.clone()
        

#     # import pdb; pdb.set_trace()
#     return d_rpr

def find_futl(d_fls):

    t_scr, t_fgm = d_fls['t_scr_6'].to(device), d_fls['t_fgm_6'].to(device)
    t_fsc   = 7*(2*(t_scr[:,-1] % 2)-1)
    t_futl  = t_fsc[t_fgm[:,1]].to(torch.float32)
    return t_futl


def find_utl(i_hnd, t_val, b_flg, d_fls,  dc_edg, dt_edg, tr_sgm, tf_sgms, d_regs, d0_rpr, d1_rpr, alg, TOL, i_cnt):

    # d_regs = {f't_reg_{i_cod}': torch.zeros(d_fls[f't_edg_{i_cod}'].shape[0] , dtype = torch.float64, device=device) for i_cod in i_cods}
   
    for i_trn in reversed(range(4)):
        for i_ply in reversed(range(2)):

            i_cod  = f'{i_hnd}_{i_trn}_{i_ply}'

            if i_ply == 1:   
                
                t_ply  = d0_rpr[i_cod] 
                t_opp  = d1_rpr[i_cod]

            else:            
                
                t_ply  = d1_rpr[i_cod] 
                t_opp  = d0_rpr[i_cod]

            c_edg, t_edg  = dc_edg[f'c_edg_{i_cod}'], dt_edg[f't_edg_{i_cod}']
            i_sid  = d_fls[f'i_sid_{i_cod}'] 
            # c_edg, t_edg, t_sgm  = d_fls[f'c_edg_{i_cod}'], d_fls[f't_edg_{i_cod}'], tr_sgm[f't_sgm_{i_cod}']
            # i_sid, d_fls_c_edg, d_fls_t_edg
            t_sgm         = tr_sgm[f't_sgm_{i_cod}']
            t_tmp         = t_sgm*t_val
            t_pt2         = torch.zeros(i_sid, dtype = torch.float32, device=device)  #TODO:64
            t_pt2.scatter_add_(0, t_edg, t_tmp) 
            t_reg         = (1-2*i_ply)*t_opp*(t_val-t_pt2[t_edg])

            t_val         =  t_pt2.clone()

            t_msk = d_regs[f't_reg_{i_cod}']>=TOL
            # t_fct = torch.zeros_like(d_regs[f't_reg_{i_cod}']).to(torch.float32) #TODO:64
            t_fct = torch.zeros(d_regs[f't_reg_{i_cod}'].shape, dtype=torch.float32, device=d_regs[f't_reg_{i_cod}'].device)
            t_fct[t_msk] = i_cnt**1.5/(1+i_cnt**1.5)
            t_fct[~t_msk] = torch.tensor(0.7, device=device).to(torch.float32) #TODO:64
            if alg == 'cfr':    d_regs[f't_reg_{i_cod}']  = t_fct*d_regs[f't_reg_{i_cod}'] + t_reg
            elif alg == 'cfr+': d_regs[f't_reg_{i_cod}']  = torch.clamp(d_regs[f't_reg_{i_cod}']+t_reg, min=0.00)
            
            else:
                NotImplementedError()

            t_sum         = torch.zeros(i_sid, dtype = torch.float32, device=device)   #TODO:64
            t_rts = torch.clamp(1000*d_regs[f't_reg_{i_cod}'], min=TOL).to(torch.float32) #TODO:64
            # import pdb; pdb.set_trace()
            t_sum.scatter_add_(0, t_edg, t_rts)
            t_msk          = t_sum[t_edg] <= TOL
            t_sgm[~t_msk]  = t_rts[~t_msk]/t_sum[t_edg][~t_msk]
            t_sgm[t_msk]   = (1/c_edg[t_edg])[t_msk].to(torch.float32) #TODO:64
            
            assert ~torch.isnan(t_sgm).any()
            if b_flg:
                if   alg == 'cfr' : tf_sgms[f't_sgm_{i_cod}']  = ((1-1/(1+i_cnt))**2)*tf_sgms[f't_sgm_{i_cod}']+t_sgm*t_ply
                elif alg == 'cfr+': tf_sgms[f't_sgm_{i_cod}']  += (1+i_cnt)*t_sgm*t_ply
            
            tr_sgm[f't_sgm_{i_cod}']   =  t_sgm.clone()

        
    return t_val, tf_sgms,  tr_sgm, d_regs

if __name__ == "__main__":

    t_dks = torch.load(f"decks_10000.pt")
    def run_cfr(i_dck, LOG_TOL, I_TER):

        TOL = 10**(-LOG_TOL)
        
        alg = 'cfr'
        t_dck = t_dks[i_dck,:].to(device)
        d_fls, t_cmps, t_fgms, t_scrs, t_m52s  = simulategamerun(t_dck.clone())
        # import pdb; pdb.set_trace()
        # d_fls = {k: v.cuda() for k, v in d_fls.items()}
        tr_sgm = {f't_sgm_{i_cod}': d_fls[f't_sgm_{i_cod}'].to(dtype=torch.float32, device=device) for i_cod in i_cods} #TODO:64
        tf_sgms  = {f't_sgm_{i_cod}': d_fls[f't_sgm_{i_cod}'].to(dtype=torch.float32, device=device) for i_cod in i_cods} #TODO:64
        [d_fls.pop(f't_sgm_{i_cod}', None) for i_cod in i_cods]
        # del d_fls[f't_sgm_{i_cod}'] for i_cod in i_cods
        t_futl   = find_futl(d_fls).to(torch.float32)
        # i_bar   = trange(I_TER, desc=f"Processing {alg}.  LOG_TOL = {LOG_TOL}")
        # for i_cnt in i_bar:

        #     for i_hnd in reversed(range(6)):
        # import pdb; pdb.set_trace()
        d_regs = {f't_reg_{i_cod}': torch.zeros(d_fls[f't_edg_{i_cod}'].shape[0] , dtype = torch.float64, device=device) for i_cod in i_cods}

        for i_hnd in reversed(range(6)):

            # print(t_futl.shape[0])
            i_bar   = trange(I_TER, desc=f"Processing {alg}. Hand {i_hnd}. t_futl.shape[0] = {t_futl.shape[0]:7.0f}")
            i_h = d_fls[f't_fgm_{i_hnd -1}'] if i_hnd > 0 else 1
            T_UTL = torch.zeros(i_h, device=device)
            T_UTLL = torch.zeros((i_h, I_TER), device=device)
            t_lnk  = d_fls[f't_lnk_{i_hnd}'].to(device=device) 
            t_rus  = d_fls[f't_rus_{i_hnd}'].to(dtype=torch.float32, device=device)  #TODO:64
            t_val  = t_futl[t_lnk] + t_rus

            dc_edg = {f'c_edg_{i_cod}': d_fls[f'c_edg_{i_cod}'].to(device=device) for i_cod in i_cods if int(i_cod.split('_')[0])==i_hnd}
            dt_edg = {f't_edg_{i_cod}': d_fls[f't_edg_{i_cod}'].to(device=device) for i_cod in i_cods if int(i_cod.split('_')[0])==i_hnd}
            

            for i_cnt in i_bar:

                # start_time = time.time()

                d0_rpr, d1_rpr = find_reach_probs(d_fls, dt_edg, tr_sgm, 0, i_hnd), find_reach_probs(d_fls, dt_edg, tr_sgm,1, i_hnd)
                # find_utl(i_hnd, t_val, b_flg, i_sid, c_edg, t_edg, tr_sgm, tf_sgms, d_regs, d0_rpr, d1_rpr, alg, TOL, i_cnt)
                # t_utl, tf_sgms,  tr_sgm = find_utl(i_hnd, t_val,  i_cnt > 0, d_fls, tr_sgm, tf_sgms, d0_rpr, d1_rpr, alg, TOL, i_cnt)
                t_utl, tf_sgms,  tr_sgm, d_regs = find_utl(i_hnd, t_val, i_cnt > 0, d_fls, dc_edg, dt_edg, tr_sgm, tf_sgms, d_regs, d0_rpr, d1_rpr, alg, TOL, i_cnt)
                T_UTL += t_utl
                T_UTLL[:,i_cnt] = T_UTL/(i_cnt+1)

                # if time.time() - start_time > 1:
                #     print(f"Exiting early for {i_dck}")
                #     return 


            
            t_futl = T_UTL/I_ITER
                
              
        # t_rprs  = find_reach_probs3(d_fls, tr_sgm)

        t_sgms, ddf_dict  = {}, {}
        t_dlt = torch.zeros((1,52), device=device)
        d_fls = {k: v.cuda() if torch.is_tensor(v) else v for k, v in d_fls.items()}

        gc.collect();torch.cuda.empty_cache()

        for i_cod in i_cods:

            i_sid         = d_fls[f'i_sid_{i_cod}']
            t_sum         = torch.zeros(i_sid, dtype = torch.float32, device=device)   #TODO:64
            c_edg, t_edg, t_sgm  = d_fls[f'c_edg_{i_cod}'], d_fls[f't_edg_{i_cod}'],  tf_sgms[f't_sgm_{i_cod}']
            t_sum.scatter_add_(0, t_edg, t_sgm)
            t_msk         = t_sum[t_edg] <= TOL
            t_sgm[~t_msk] = t_sgm[~t_msk]/t_sum[t_edg][~t_msk]
            t_sgm[t_msk]   = (1/c_edg[t_edg])[t_msk].to(torch.float32)  #TODO:64
            t_sgms[i_cod] = t_sgm.cpu()


            

        gc.collect();torch.cuda.empty_cache()
        t_cmps = {k: v.cuda() for k, v in t_cmps.items()}
        t_fgms = {k: v.cuda() for k, v in t_fgms.items()}
        t_scrs = {k: v.cuda() for k, v in t_scrs.items()}
        for i_hnd in i_hnds:
                
            t_scr = t_scrs[i_hnd][:,[0,1,3]]
            t_m52 = t_m52s[i_hnd]

            if i_hnd > 0:
            
                i_nnd = 8*i_hnd+4
                t_dlt[0,t_dck[:i_nnd]] = 1
            
            for i_trn in range(4):
                for i_ply in range(2):

                    i_cod = f'{i_hnd}_{i_trn}_{i_ply}'


                    t_sgm = t_sgms[i_cod]
                    t_cmp = t_cmps[i_cod] 
                    t_fgm = t_fgms[i_cod] 
                    # t_rpr = t_rprs[i_cod]
                    
                    t_prq = torch.zeros((t_fgm.shape[0], 60), dtype=torch.int8, device=device)
                    t_prq[:,t_m52]       = t_cmp[t_fgm[:,0]]
                    t_prq[:,52:55]       = t_scr[t_fgm[:,1]] #52,53,54
                    t_prq[:,55:58]       = torch.tensor([i_hnd,i_trn,i_ply], dtype=torch.int8, device=device) #55,56,57

                    t_prq[torch.logical_and(F.pad(t_dlt, (0, 8))==1, t_prq==0)] = -127
                    t_prq                = t_prq.cpu()

                    data = {get_col(i): t_prq[:, i] for i in range(t_prq.shape[1])}
                    df   = pd.DataFrame(data)

                    # df["D"]   = df["D"].astype("int32")
                    # if isinstance(i_dck, str):
                    #     df["D"]   = np.full(len(df), -int(i_dck[:-1]), dtype=np.int32)
                    # else:
                    #     df["D"]   = np.full(len(df), i_dck, dtype=np.int32)
                    df['SGM'] = torch.clamp(t_sgm.to(torch.float16), min=0.0, max=1.0)
                    df['D']  = np.full(len(df), i_dck, dtype=np.int32)
                    # df["RPO"] = t_rpr.to(device='cpu', dtype=torch.float16)
                
                    ddf = dd.from_pandas(df, npartitions=1)
                    ddf_dict[i_cod] = ddf

        ddf_all = dd.concat(list(ddf_dict.values())).repartition(npartitions=1)
        ddf_all.to_parquet(
            f"../PRQ2/D{i_dck}_{I_ITER}_{LOG_TOL}",
            engine="pyarrow",
            write_index=False,
            write_metadata_file=False  # optional, suppress _metadata if you want
        )

    ALL_LOGS  = [12] 
    ALL_ITERS = [1000]


    
    # for tr in range(1):
    for i_dck in range(10):
        
        for LOG_TOL in ALL_LOGS:
            for I_ITER in ALL_ITERS:
                run_cfr(i_dck, LOG_TOL,I_ITER)