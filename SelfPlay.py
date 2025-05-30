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


def playrandom(t_dck, N=1, to_latex = False):


    t_m52 = torch.tensor([True for i in range(52)], device=device)
    find_dm = lambda lm :  torch.tensor([idx in lm for idx in torch.arange(52, device=device)], dtype = bool, device=device)
    find = lambda t_s, t_l: (t_s[:, None, :] == t_l[None, :, :]).all(dim=2).float().argmax(dim=1)

    d_msk = {}
    d_msk['cur_k'] = find_dm(torch.arange(48,52, device=device))
    d_msk['cur_q'] = find_dm(torch.arange(44,48, device=device))
    d_msk['cur_j'] = find_dm(torch.arange(40,44, device=device))
    d_msk['cur_n'] = find_dm(torch.arange(40, device=device))
    d_msk['cur_c'] = find_dm(4*torch.arange(13, device=device))
    d_msk['cur_p'] = find_dm(torch.tensor([0,1,2,3,4,37,40,41,42,43], device=device))
    d_msk['cur_s'] = torch.tensor([1,1,1,1,2,3,1,1,1,1],dtype = INT8, device=device)
    d_msk['cur_52']= t_m52

    i_pad_n, _       = pad_helper(d_msk, 'n') 
    i_pad_k, t_pad_k = pad_helper(d_msk, 'k')
    i_pad_q, t_pad_q = pad_helper(d_msk, 'q')
    i_pad_j, _       = pad_helper(d_msk, 'j')
    d_pad = {'i_pad_n': i_pad_n, 'i_pad_k':i_pad_k, 'i_pad_q':i_pad_q, 'i_pad_j':i_pad_j, 't_pad_k':t_pad_k, 't_pad_q':t_pad_q}

    t_ltx   = {}

    
    
    # t_dck   = t_dck.repeat(N,1)
    clt_dck = t_dck.clone()
    t_inf   = torch.zeros((N,3,52), dtype=INT8, device=device)
    # t_idx   = torch.arange(t_inf.shape[0], device=t_dck.device).unsqueeze(1).expand(-1, 4)
    # t_inf[t_idx,0,t_dck[:,:4]] = 3
    t_inf[:,0,t_dck[:4]] = 3

    t_dck  = t_dck[4:]
    # t_idx  = torch.arange(t_inf.shape[0], device=t_dck.device).unsqueeze(1).expand(-1, 4)
    t_nnd  = torch.zeros((1,52), dtype =torch.float32, device=device)
    t_scr  = torch.zeros((t_inf.shape[0],len(d_scr)), device=device,dtype=INT8)
     

    for i_hnd in range(6):

        t_snw     = torch.zeros((t_inf.shape[0],len(d_snw)), device=device,dtype=INT8)

        if i_hnd > 0: 
            t_nnd[:,clt_dck[:8*i_hnd+4]] = 1
        # torch.cuda.empty_cache()
        t_inf[:,0, t_dck[:4]]  = 1
        t_inf[:,0, t_dck[4:8]] = 2
        t_dck                        = t_dck[8:]
        
        if to_latex: t_ltx[f't_scr_{i_hnd}'] = t_scr.squeeze(0).clone()

        for i_trn in range(4):
            for i_ply in range(2):
                
                i_cod = f'{i_hnd}_{i_trn}_{i_ply}'
                print(i_cod, t_inf.shape[0])

                t_act, c_act  = find_moves(t_inf,i_ply, d_msk, d_pad) 


                if to_latex: t_ltx['t_dck_'+i_cod] = t_inf[:,0,:].clone()

                t_inf          = torch.repeat_interleave(t_inf, c_act, dim=0)
                t_snw          = torch.repeat_interleave(t_snw, c_act, dim=0)
                t_inf, t_snw   = apply_moves(t_inf, t_act, t_snw, d_msk,  i_hnd, i_ply, i_trn)

                # t_tmp              = t_inf[t_sid[:,0]]
                # t_xgl                 = t_tmp[:,1,:]-t_tmp[:,2,:]
                # t_xgl[torch.logical_and(t_tmp[:,0,:]==3, t_xgl==0)] = 110        ### Card was already in the pool 
                # t_xgl[torch.logical_and(t_tmp[:,0,:]==i_ply+1, t_xgl==0)] = 100  ### Player has the card

                # t_xgb          = torch.zeros((t_sid.shape[0],56), dtype=torch.int8, device=device)
                # t_xgb[:,torch.cat([t_m52,torch.tensor([False,False,False,False], device=device)],dim=0)] = t_xgl
                # t_xgb[torch.logical_and(F.pad(t_nnd, (0, 4))==1, t_xgb==0)] = -127

                t_xgl                 = t_inf[:,1,:]-t_inf[:,2,:]
                t_xgl[torch.logical_and(t_inf[:,0,:]==3, t_xgl==0)] = 110  ## already in pool
                t_xgl[torch.logical_and(t_inf[:,0,:]==i_ply+1, t_xgl==0)] = 100 ## holds

                t_xgb          = torch.zeros((t_inf.shape[0],56), dtype=torch.int8, device=device)
                t_xgb[:,torch.cat([t_m52,torch.tensor([False,False,False,False], device=device)],dim=0)] = t_xgl
                t_xgb[torch.logical_and(F.pad(t_nnd, (0, 4))==1, t_xgb==0)] = -127
                t_xgb[:,52:] = torch.repeat_interleave(t_scr, c_act, dim=0)
                

               
                i_max         = c_act.max()
                t_msk         = torch.arange(i_max, device=device).unsqueeze(0) < c_act.unsqueeze(1)
                t_mtx         = torch.zeros_like(t_msk, device=device, dtype=torch.float32)
                t_mtx[t_msk]  = torch.tensor(1,dtype=torch.float32, device=device) 
                t_smp         = torch.multinomial(t_mtx, num_samples=1).squeeze(1) 

                t_gps         = torch.cat([torch.tensor([0], device=device), c_act.cumsum(0)[:-1]]) # group starts 
                m_exs         = torch.zeros(t_inf.shape[0], dtype=torch.bool, device=device)
                
                m_exs[t_gps + t_smp] = True 
                
                t_inf         = t_inf[m_exs]
                t_snw         = t_snw[m_exs]
               

               
                if to_latex:

                    t_ltx['t_act_'+i_cod] =  t_act[m_exs].clone()
                    t_ltx['t_snw_'+i_cod] =  t_snw.squeeze(0).clone()

        

        t_snw[:,d_snw['pts_dlt']]   = t_snw[:,d_snw['a_pts']] + t_snw[:,d_snw['a_sur']] - t_snw[:,d_snw['b_pts']] - t_snw[:,d_snw['b_sur']]
        
        t_scr[:,:3] = t_scr[:,:3]+t_snw[:,:3].clone()
        t_mal = t_scr[:,d_scr['a_clb']]>=7
        t_mbb = t_scr[:,d_scr['b_clb']]>=7
        t_scr[:,d_scr['max_clb']][t_mal]= 1
        t_scr[:,d_scr['max_clb']][t_mbb]= 2
        t_mcl = t_scr[:,d_scr['max_clb']]>0
        t_scr[:,0:2].masked_fill_(t_mcl.unsqueeze(-1), 0)
        if i_hnd ==  5: 
            t_inf, t_snw = cleanpool(t_inf, t_snw, d_msk)
            if to_latex: 
                t_ltx['t_snw_6'] = t_snw.squeeze(0).clone()


        t_inf        = t_inf[:,0,:].unsqueeze(1)
        t_inf = F.pad(t_inf, (0, 0, 0, 2))


    t_fsc = 7*(2*(t_scr[:,-1] % 2)-1)+t_scr[:,-2]
    t_win = (t_fsc>=0).sum()/t_fsc.shape[0]
    return t_win, t_ltx



if __name__ == "__main__":

    t_dck = torch.tensor([47, 40, 45, 16, 23, 20,  1,  4, 21, 17, 36, 29, 18, 32, 49,  9, 34, 24,
        37, 41, 48, 44, 27,  8, 51,  2,  6,  0, 11, 10, 26, 33, 12, 42, 15, 13,
            3, 50, 39, 35, 43, 28,  7, 46, 30, 22, 19,  5, 25, 38, 31, 14],
        device='cuda:0')

    t0_win, ltx = playrandom(t_dck, N=100, to_latex=True)







    

  
    
