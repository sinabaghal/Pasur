import torch, sys, os
import torch.nn.functional as F
from Setup import setup
from Utils import get_d_mask, pad_helper
from FindMoves import find_moves
from ApplyMoves import apply_moves
from Imports import device, INT8, INT32, d_snw, d_scr
from RepeatedBlocks import repeatedblocks
from CleanPool import cleanpool 
from ExternalSampling import extsampling, sorttsid
from zstd_store import to_memory
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt


id_keys = [
            f'{i_hnd}_{i_trn}_{i_ply}'
            for i_hnd in range(6)
            for i_trn in range(4)
            for i_ply in range(2)
            ]

def simulategamerun():
   
 
    d_fls = {}
    t_nns   = {}
    
    t_dck = torch.tensor([47, 40, 45, 16, 23, 20,  1,  4, 21, 17, 36, 29, 18, 32, 49,  9, 34, 24,
        37, 41, 48, 44, 27,  8, 51,  2,  6,  0, 11, 10, 26, 33, 12, 42, 15, 13,
         3, 50, 39, 35, 43, 28,  7, 46, 30, 22, 19,  5, 25, 38, 31, 14],
       device='cuda:0')
    
    t_m52 = torch.tensor([True if i in t_dck[:4] else False for i in range(52)], device=device)
    t_mdd = torch.tensor([False for _ in range(52)], device=device)
    
    t_inf        = torch.zeros((1,3,4), dtype=INT8, device=device)
    t_inf[0,0,:] = 3
    
    clt_dck = t_dck.clone()
    t_dck   = t_dck[4:]

    t_scr   = torch.zeros((1,len(d_scr)), device=device,dtype=INT8)
    t_sid   = torch.zeros((1,2), device=device,dtype=torch.int32)


    for i_hnd in range(6):

        t_nnd = torch.zeros((1,52), device=device)
        if i_hnd > 0:
            
            i_nnd = 8*i_hnd+4
            t_nnd[0,clt_dck[:i_nnd]] = 1


        torch.cuda.empty_cache()
        _, c_scr = torch.unique(t_sid[:,0],dim=0,return_counts=True)

        d_msk = get_d_mask(t_m52, t_dck[:8])
        t_dck = t_dck[8:]
        t_inf = F.pad(t_inf, (0, 8))[:,:,d_msk['pad']]
        t_inf[:,0, d_msk['nxt_a']] = 1
        t_inf[:,0, d_msk['nxt_b']] = 2
        i_pad_n, _       = pad_helper(d_msk, 'n') 
        i_pad_k, t_pad_k = pad_helper(d_msk, 'k')
        i_pad_q, t_pad_q = pad_helper(d_msk, 'q')
        i_pad_j, _       = pad_helper(d_msk, 'j')

        d_pad = {'i_pad_n': i_pad_n, 'i_pad_k':i_pad_k, 'i_pad_q':i_pad_q, 'i_pad_j':i_pad_j, 't_pad_k':t_pad_k, 't_pad_q':t_pad_q}
        t_snw = torch.zeros((t_inf.shape[0],len(d_snw)), device=device,dtype=INT8)
    
        for i_trn in range(4):
            for i_ply in range(2):
                
                i_cod = f'{i_hnd}_{i_trn}_{i_ply}'
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
                
                t_tmp              = t_inf[t_sid[:,0]]

                t_xgl                 = t_tmp[:,1,:]-t_tmp[:,2,:]
                t_xgl[torch.logical_and(t_tmp[:,0,:]==3, t_xgl==0)] = 110        ### Card was already in the pool 
                t_xgl[torch.logical_and(t_tmp[:,0,:]==i_ply+1, t_xgl==0)] = 100  ### Player has the card
                t_xgl[torch.logical_and(t_nnd.squeeze(0)[t_m52] == 1, t_xgl==0)] = -127

                t_xgb          = torch.zeros((t_sid.shape[0],56), dtype=torch.int8, device=device)
                t_xgb[:,torch.cat([t_m52,torch.tensor([False,False,False,False], device=device)],dim=0)] = t_xgl
                t_xgb[:,52:] = t_scr[t_sid[:,1]]
               
                t_sgm = 1/torch.repeat_interleave(c_edg, c_edg)

                d_fls[f'i_sid_{i_cod}'] = i_sid
                d_fls[f't_sgm_{i_cod}'] = t_sgm
                d_fls[f't_edg_{i_cod}'] = t_edg
                d_fls[f'c_edg_{i_cod}'] = c_edg
                
                t_nns[i_cod]  = t_xgb
          
        if i_hnd ==  5:  t_inf, t_snw = cleanpool(t_inf, t_snw, d_msk)

        t_inf        = t_inf[:,0,:]
        t_snw[:,d_snw['pts_dlt']]   = t_snw[:,d_snw['a_pts']] + t_snw[:,d_snw['a_sur']] - t_snw[:,d_snw['b_pts']] - t_snw[:,d_snw['b_sur']]
        t_snw        = t_snw[:,:4]
        t_snw[:,-1]  = 0    
        t_inf, t_lnk = torch.unique(t_inf,dim=0, sorted=False, return_inverse=True)
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
        t_ndd        = torch.logical_and(t_c52,~t_m52)
        t_mdd[t_ndd] = True
        t_inf        = t_inf.unsqueeze(1)
        
        if t_inf.shape[2] == 0:
            t_inf = torch.empty((t_inf.shape[0], 3, 0), device='cuda:0', dtype=INT8)
        else:
            t_inf = F.pad(t_inf, (0, 0, 0, 2))

    d_fls['t_scr_6'], d_fls['t_sid_6'] = t_scr, t_sid

    # to_memory(torch.cat([t_nns[id] for id in id_keys], axis=0),  f"../DATA/nns/tensors.pt.zst")

    return d_fls

def selfplay(N=100, to_latex = False, d_fls = {}):


    t_m52 = torch.tensor([True for i in range(52)], device=device)
    find_dm = lambda lm :  torch.tensor([idx in lm for idx in torch.arange(52, device=device)], dtype = bool, device=device)

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

  
        
    t_dck = torch.tensor([47, 40, 45, 16, 23, 20,  1,  4, 21, 17, 36, 29, 18, 32, 49,  9, 34, 24,
        37, 41, 48, 44, 27,  8, 51,  2,  6,  0, 11, 10, 26, 33, 12, 42, 15, 13,
            3, 50, 39, 35, 43, 28,  7, 46, 30, 22, 19,  5, 25, 38, 31, 14],
        device='cuda:0')
    
    t_dck  = t_dck.repeat(N,1)
    clt_dck = t_dck.clone()
    t_inf   = torch.zeros((t_dck.shape[0],3,52), dtype=INT8, device=device)
    t_idx = torch.arange(t_inf.shape[0], device=t_dck.device).unsqueeze(1).expand(-1, 4)
    t_inf[t_idx,0,t_dck[:,:4]] = 3

    t_dck = t_dck[:,4:]
    t_idx = torch.arange(t_inf.shape[0], device=t_dck.device).unsqueeze(1).expand(-1, 4)
    t_nnd = torch.zeros((t_inf.shape[0],52), dtype =torch.float32, device=device)
    t_scr = torch.zeros((t_inf.shape[0],len(d_scr)), device=device,dtype=INT8)

    for i_hnd in range(6):

        t_snw     = torch.zeros((t_inf.shape[0],len(d_snw)), device=device,dtype=INT8)

        if i_hnd > 0: 
            t_nnd[:,clt_dck[:8*i_hnd+4]] = 1
        # torch.cuda.empty_cache()
        t_inf[t_idx,0, t_dck[:,:4]]  = 1
        t_inf[t_idx,0, t_dck[:,4:8]] = 2
        t_dck                        = t_dck[:,8:]
        
        if to_latex: t_ltx[f't_scr_{i_hnd}'] = t_scr.squeeze(0).clone()

        for i_trn in range(4):
            for i_ply in range(2):
                
                i_cod = f'{i_hnd}_{i_trn}_{i_ply}'

                t_act, c_act  = find_moves(t_inf,i_ply, d_msk, d_pad) 
                if to_latex:
                    t_ltx['t_dck_'+i_cod]    = t_inf[:,0,:].clone()
            

                t_inf          = torch.repeat_interleave(t_inf, c_act, dim=0)
                t_snw          = torch.repeat_interleave(t_snw, c_act, dim=0)
                t_inf, t_snw   = apply_moves(t_inf, t_act, t_snw, d_msk,  i_hnd, i_ply, i_trn)
                t_xgl                 = t_inf[:,1,:]-t_inf[:,2,:]
                t_xgl[torch.logical_and(t_inf[:,0,:]==3, t_xgl==0)] = 110
                t_xgl[torch.logical_and(t_inf[:,0,:]==i_ply+1, t_xgl==0)] = 100
                t_xgl[torch.logical_and(torch.repeat_interleave(t_nnd, c_act,dim=0)==1, t_xgl==0)] = -127
                t_xgb          = torch.zeros((t_inf.shape[0],56), dtype=torch.int8, device=device)
                t_xgb[:,torch.cat([t_m52,torch.tensor([False,False,False,False], device=device)],dim=0)] = t_xgl

                t_xgb[:,52:] = torch.repeat_interleave(t_scr, c_act, dim=0)


                if i_ply == 0:

                    t_sgm = d_fls[f't_sgm_{i_cod}'].unsqueeze(0).repeat(N, 1).reshape(-1)
                    
                else:
                    t_sgm = torch.tensor(1, dtype= torch.float32, device=device)



                i_max         = c_act.max()
                t_msk         = torch.arange(i_max, device=device).unsqueeze(0) < c_act.unsqueeze(1)
                t_mtx         = torch.zeros_like(t_msk, dtype=t_sgm.dtype)
                t_mtx[t_msk]  = t_sgm
                t_smp         = torch.multinomial(t_mtx, num_samples=1).squeeze(1) 
                t_gps         = torch.cat([torch.tensor([0], device=device), c_act.cumsum(0)[:-1]]) # group starts 
                
                t_exs         = torch.zeros(t_inf.shape[0], dtype=torch.bool, device=device)
                t_exs[t_gps + t_smp] = True 

                t_inf         = t_inf[t_exs]
                t_snw         = t_snw[t_exs]
            
            
                if to_latex:

                    t_ltx['t_act_'+i_cod] =  t_act[t_exs].clone()
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

    d_fls = simulategamerun()

    t_scr, t_sid = d_fls['t_scr_6'], d_fls['t_sid_6']
    t_fsc = 7*(2*(t_scr[:,-1] % 2)-1)+t_scr[:,-2]
    t_vll = t_fsc[t_sid[:,1]].to(torch.float32)

    cnt = 0
    while cnt < 100:

       
        t_val = t_vll

        for i_hnd in reversed(range(6)):

            t_lnk  = d_fls[f't_lnk_{i_hnd}'] 
            t_val  = t_val[t_lnk]
            
            for i_trn in reversed(range(4)):
                for i_ply in reversed(range(2)):


                    i_cod        = f'{i_hnd}_{i_trn}_{i_ply}'
                    i_sid        = d_fls[f'i_sid_{i_cod}'] 
                    t_sum        = torch.zeros(i_sid, dtype = t_val.dtype, device=device)  
                    t_edg, c_edg, t_sgm = d_fls[f't_edg_{i_cod}'],  d_fls[f'c_edg_{i_cod}'], d_fls[f't_sgm_{i_cod}']
                    t_sum.scatter_add_(0, t_edg, t_sgm*t_val)
                    
                    t_reg         = t_val - t_sum[t_edg]
                    t_val         = t_sum
                    t_reg         = torch.clamp(t_reg, min=0)
                    
                    # os.makedirs(f"../FIGS/{i_cod}/", exist_ok=True)
                    # plt.hist(t_reg.cpu().numpy(), bins=30); plt.savefig(f"../FIGS/{i_cod}/{cnt}.png");plt.clf()

                    t_sum         = torch.zeros(i_sid, dtype = t_val.dtype, device=device)  
                    t_sum.scatter_add_(0, t_edg, t_reg)
                    
                    t_sgm         = torch.zeros(t_sgm.shape[0], dtype = t_sum.dtype, device=device)
                    t_msk         = t_sum[t_edg] == 0 
                    t_szr         = t_sum == 0
                    t_sgm[t_msk]  = 1/torch.repeat_interleave(c_edg[t_szr], c_edg[t_szr])
                    t_sgm[~t_msk] = (t_reg[~t_msk]/t_sum[t_edg][~t_msk])

                    d_fls[f't_sgm_{i_cod}'] = t_sgm 

        
        cnt +=  1
    

    t_win, _ = selfplay(N=100, to_latex = False, d_fls = d_fls)
    import pdb; pdb.set_trace()
    # to_memory(torch.cat([d_fls[f't_sgm_{id}'] for id in id_keys], axis=0),  f"../DATA/nns/sigmas.pt.zst")







    