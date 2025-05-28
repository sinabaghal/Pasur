
import torch.nn.functional as F
from Actions import n_actions, j_actions, k_actions
from Utils import expand_unique, pad_helper, print_deck, find_tz_deck
import torch 
import os, gzip
from ZipTools import to_zstd


from Imports import device, INT8, INT32, d_snw

def play_round(i_hnd, seed, t_inf, t_sid, dm, t_scr, t_dck, t_m52, t_mdd, t_hlp):

    
    t_40x2764, t_2764x40, t_tuples, D_S2T = t_hlp
    torch.cuda.empty_cache()

    t_snw = torch.zeros((t_inf.shape[0],len(d_snw)), device=device,dtype=INT8)
    
    tm_p     = dm['cur_p']
    tm_s     = dm['cur_s']
    tm_c     = dm['cur_c']
    tm_n     = dm['cur_n']
    tm_j     = dm['cur_j']
    tm_q     = dm['cur_q']
    tm_k     = dm['cur_k']
    t_m52    = dm['cur_52']
    i_pad_n, _       = pad_helper(dm, 'n') 
    i_pad_k, t_pad_k = pad_helper(dm, 'k')
    i_pad_q, t_pad_q = pad_helper(dm, 'q')
    i_pad_j, _       = pad_helper(dm, 'j')

    



    def apply_moves(t_inf, t_act, t_snw, i_ply, i_trn):
        
        t_mpk             = torch.any(t_act[:,1,:],dim=1) 
        tm_pick_notj      =  torch.any(t_act[:,0,tm_j],dim=1) == 0
        t_inf[~t_mpk,0,:] += (2-i_ply)*t_act[~t_mpk,0,:] ## add to pool
        t_inf[t_mpk,0,:]  -= (1+i_ply)*t_act[t_mpk,0,:] ## remove to pool
        t_inf[t_mpk,0,:]  -= 3*t_act[t_mpk,1,:]   ## remove from pool
        t_inf[:,i_ply+1,:]    += (i_trn+1)*t_act[:,0,:]+10*(i_trn+1)*t_act[:,1,:] # v*card held by player + 10*v*cards picked from the pool  

        ## apply_moves score tensor
        plyr = 'a' if i_ply ==0 else 'b'
        clubs_cntr, pt_cntr, surs_cntr = d_snw[f'{plyr}_clb'], d_snw[f'{plyr}_pts'],d_snw[f'{plyr}_sur']
        # clubs_cntr, pt_cntr = d_snw[f'{plyr}_clb'], d_snw[f'{plyr}_pts']

        
        t_clt                    = t_act[t_mpk, 1, :]+t_act[t_mpk, 0, :]
        t_snw[t_mpk,clubs_cntr] += t_clt[:,tm_c].sum(dim=1).to(INT8)
        t_snw[t_mpk,pt_cntr]    += (t_clt[:,tm_p]*tm_s).sum(dim=1).to(INT8)
        
        t_snw[t_mpk,d_snw['lst_pck']] = i_ply+1

        if i_hnd < 5:
            t_sur = torch.any(t_inf[:, 0, :],dim=1) == 0 
            t_msr = torch.logical_and(torch.logical_and(t_mpk, tm_pick_notj), t_sur)
            # if torch.any(t_msr):
            #     import pdb; pdb.set_trace()
            t_snw[t_msr,surs_cntr] += 5
        
        assert t_inf.dtype == INT8
        return t_inf, t_snw


    t_inds = torch.arange(t_inf.shape[0]+1, device=device)

    
    for i_trn in range(4):
        for i_ply in range(2):

            torch.cuda.empty_cache()
            i_cod = f'{i_trn}_{i_ply}'
        
            assert t_inf.dtype == torch.int8
            t_act, c_act  = find_moves(t_inf,i_ply)
           
            to_zstd(t_act,  i_cod)
            to_zstd(c_act, i_cod)
           
            tc_scr_full = torch.repeat_interleave(c_scr, c_act, dim=0)
            t_z_full    = torch.repeat_interleave(t_inf, c_act, dim=0)
            tc_scr_full = torch.repeat_interleave(c_scr, c_act, dim=0)

            # to_zstd(tc_scr_full, tc_scr_file)
            to_zstd(t_z_full, i_cod)
            
            if i_ply == seed%2:
                
                tc_act_smpl = torch.floor(torch.rand_like(c_act, dtype=torch.float) * c_act).to(INT32) 
                tc_act_cumsum = c_act.cumsum(0)
                tc_act_cumsum = torch.cat([torch.tensor([0], device=device),tc_act_cumsum[:-1]])
                t_indices_smpl = tc_act_smpl+tc_act_cumsum
                t_act = t_act[t_indices_smpl]
                c_act = torch.ones_like(c_act, device=device, dtype=INT32)
                t_inf    = torch.repeat_interleave(t_inf, c_act, dim=0)
                # c_scr = tc_scr_full[t_indices_smpl]
                to_zstd(c_scr, tc_scr_file)
                assert t_act.shape[0] == c_act.shape[0]
                t_cumsum      = torch.zeros(t_inds.shape[0]-1, device=device, dtype=INT32)
                t_indices     = torch.arange(t_inds.shape[0]-1, device=device).repeat_interleave(t_inds[1:]-t_inds[:-1])
                t_cumsum.scatter_add_(0, t_indices, c_act)
                assert t_cumsum.dtype == INT32
                to_zstd(t_cumsum, t_cumsum_file)
                t_inds         = torch.cat((torch.tensor([0], device=device),torch.cumsum(t_cumsum, dim=0)),dim=0)
            else:
                
                t_inf   = t_z_full
                c_scr = tc_scr_full
                to_zstd(tc_scr_full, tc_scr_file)
                t_cumsum      = torch.zeros(t_inds.shape[0]-1, device=device, dtype=INT32)
                t_indices     = torch.arange(t_inds.shape[0]-1, device=device).repeat_interleave(t_inds[1:]-t_inds[:-1])
                t_cumsum.scatter_add_(0, t_indices, c_act)
                assert t_cumsum.dtype == INT32
                to_zstd(t_cumsum, t_cumsum_file)
                t_inds         = torch.cat((torch.tensor([0], device=device),torch.cumsum(t_cumsum, dim=0)),dim=0)
                
            
            
            t_snw      = torch.repeat_interleave(t_snw,c_act,dim=0)
            t_inf, t_snw = apply_moves(t_inf, t_act, t_snw, i_ply, i_trn)


                # tf_act_file    = f"{folder}\\tf{i_hnd}_act_{i_trn}_{i_ply}.pt"
                # tfc_act_file   = f"{folder}\\tfc{i_hnd}_act_{i_trn}_{i_ply}.pt"
                # tf_z_file      = f"{folder}\\tf{i_hnd}_z_{i_trn}_{i_ply}.pt"
                # tf_cumsum_file = f"{folder}\\tf{i_hnd}_cumsum_{i_trn}_{i_ply}.pt"
                # tfc_scr_file   = f"{folder}\\tfc{i_hnd}_scr_{i_trn}_{i_ply}.pt"

                # to_zstd(t_act,  tf_act_file)
                # to_zstd(c_act, tfc_act_file)

                # t_cumsum      = torch.zeros(t_inds.shape[0]-1, device=device, dtype=INT32)
                # t_indices     = torch.arange(t_inds.shape[0]-1, device=device).repeat_interleave(t_inds[1:]-t_inds[:-1])
                # t_cumsum.scatter_add_(0, t_indices, c_act)
                # assert t_cumsum.dtype == INT32
                # to_zstd(t_cumsum, tf_cumsum_file)
            
                # # t_edges        = torch.repeat_interleave(c_act, c_scr, dim=0)
                # tc_scr_nxt         = torch.repeat_interleave(c_scr, c_act, dim=0)
                # to_zstd(tc_scr_nxt, tfc_scr_file)
                # to_zstd(c_scr, tc_scr_file)
                # t_z_nxt            = torch.repeat_interleave(t_inf, c_act, dim=0)
                # to_zstd(t_z_nxt, tf_z_file)
                # assert t_inf.dtype == torch.int8

                
            # else:
            #     to_zstd(t_act,  t_act_file)
            #     to_zstd(c_act, tc_act_file)
            #     t_cumsum      = torch.zeros(t_inds.shape[0]-1, device=device, dtype=INT32)
            #     t_indices     = torch.arange(t_inds.shape[0]-1, device=device).repeat_interleave(t_inds[1:]-t_inds[:-1])
            #     t_cumsum.scatter_add_(0, t_indices, c_act)
            #     assert t_cumsum.dtype == INT32
            #     to_zstd(t_cumsum, t_cumsum_file)
            #     t_inds         = torch.cat((torch.tensor([0], device=device),torch.cumsum(t_cumsum, dim=0)),dim=0)
            
            #     # t_edges        = torch.repeat_interleave(c_act, c_scr, dim=0)
            #     c_scr         = torch.repeat_interleave(c_scr, c_act, dim=0)
            #     to_zstd(c_scr, tc_scr_file)
            #     t_inf            = torch.repeat_interleave(t_inf, c_act, dim=0)
            #     to_zstd(t_inf, t_z_file)
            #     assert t_inf.dtype == torch.int8
            #     t_snw      = torch.repeat_interleave(t_snw,c_act,dim=0)
            #     t_inf, t_snw = apply_moves(t_inf, t_act, t_snw, i_ply, i_trn)
                
            assert t_inf.dtype == torch.int8
            assert t_snw.dtype == torch.int8

            ## TO SAVE: t_inf, t_act, c_scr, t_edges, t_cumsum

    # import pdb; pdb.set_trace()
    if i_hnd == 5:
        
        tm_la = t_snw[:,d_snw['lst_pck']] == 1
        tm_lb = t_snw[:,d_snw['lst_pck']] == 2
        # assert tm_lb.sum()+tm_la.sum() == tm_la.shape[0]
        t_pla = t_inf[tm_la,0,:]
        t_plb = t_inf[tm_lb,0,:]
        
        t_pla[t_pla>0]=1
        t_plb[t_plb>0]=1
        
        t_inf[tm_la,1,:] += 50*t_pla
        t_inf[tm_lb,2,:] += 50*t_plb

        t_snw[tm_la,d_snw['a_clb']] += t_pla[:,tm_c].count_nonzero(dim=1).to(INT8)
        t_snw[tm_lb,d_snw['b_clb']] += t_plb[:,tm_c].count_nonzero(dim=1).to(INT8)
        t_snw[tm_la,d_snw['a_pts']] += (t_pla[:,tm_p]*tm_s).sum(dim=1).to(INT8)
        t_snw[tm_lb,d_snw['b_pts']] += (t_plb[:,tm_p]*tm_s).sum(dim=1).to(INT8)

        # del tm_la, tm_lb, t_pla, t_plb
    
    t_snw[:,d_snw['pts_dlt']]   = t_snw[:,d_snw['a_pts']] + t_snw[:,d_snw['a_sur']] - t_snw[:,d_snw['b_pts']] - t_snw[:,d_snw['b_sur']]
    
    # t_inf   = t_inf[:,0,:]
    t_snw = torch.cat((t_snw[:,:d_snw['lst_pck']],torch.zeros((t_snw.shape[0],1), device=device,dtype=INT8)),dim=1) 
    # t_snw = t_snw[:,:4]
    assert t_snw.dtype == torch.int8
    assert t_inf.dtype == torch.int8
    i_dscrd = t_mdd[[4*x for x in range(13)]].sum().item()
    assert t_snw[:,:2].sum(dim=1).max() <= 13-i_dscrd
    t_scrnn = t_sid.repeat_interleave(t_cumsum[t_sid[:,0]], dim=0)[:,1]
    return t_inf, t_snw, c_scr, t_scrnn
