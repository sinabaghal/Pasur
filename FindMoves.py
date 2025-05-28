from Actions import n_actions, j_actions, k_actions
import torch.nn.functional as F
from Utils import expand_unique
from Imports import device, INT8, INT32, t_40x2764, t_2764x40, t_tuples, D_S2T
import torch 


def find_moves(t_inf, i_ply, d_msk, d_pad):

    tm_n     = d_msk['cur_n']
    tm_j     = d_msk['cur_j']
    tm_q     = d_msk['cur_q']
    tm_k     = d_msk['cur_k']
    t_m52    = d_msk['cur_52']
    i_pad_n  = d_pad['i_pad_n']
    i_pad_k  = d_pad['i_pad_k']
    i_pad_q  = d_pad['i_pad_q']
    i_pad_j  = d_pad['i_pad_j']
    t_pad_k  = d_pad['t_pad_k']
    t_pad_q  = d_pad['t_pad_q']

   

    t_2x52 = torch.zeros((t_inf.shape[0],2,t_inf.shape[2]), dtype=INT8, device=device) 
    tm_plyr = t_inf[:,0,:] == i_ply+1
    tm_pool = t_inf[:,0,:] == 3
    t_2x52[:,0,:][tm_plyr] = 1
    t_2x52[:,1,:][tm_pool] = 1
    # import pdb; pdb.set_trace()


    ### Construct numerical actions 

    t_2x40 = t_2x52[:,:,tm_n]
    
    tu_2x40, t_inv_inds = torch.unique(t_2x40,dim=0,return_inverse=True)
    tu_pick, tu_pick_cnt,tu_lay, tu_lay_cnt = \
            n_actions(tu_2x40, t_2764x40[:,t_m52[:40]], t_40x2764[t_m52[:40],:],t_tuples) 

    tc_pick, tc_lay = tu_pick_cnt[t_inv_inds], tu_lay_cnt[t_inv_inds]
    t_pick, t_lay = tu_pick[expand_unique(tu_pick_cnt,t_inv_inds)], tu_lay[expand_unique(tu_lay_cnt,t_inv_inds)]
    ### Construct jack actions
    
    t_j, tc_j =  j_actions(t_2x44 = t_2x52[:,:,tm_n+tm_j], i_cur_n = tm_n.sum())
    # import pdb; pdb.set_trace()
    ### Construct king actions 
    
    if t_m52[48:52].sum() == 0 :
        tc_k, t_k = torch.zeros(t_2x52.shape[0],device=device,dtype=INT32), torch.zeros((0,2,t_2x52.shape[2]), device=device,dtype=INT8)
    else:
        t_2x4 = t_2x52[:,:,tm_k]
        tu_2x4, t_inv_inds = torch.unique(t_2x4,dim=0,return_inverse=True)
        tu_k, tu_k_cnt            = k_actions(tu_2x4,t_m52[48:52], D_S2T)
        tc_k, t_k                 = tu_k_cnt[t_inv_inds], tu_k[expand_unique(tu_k_cnt,t_inv_inds)]
    ### Construct queen actions 

    if t_m52[44:48].sum() == 0 :
        tc_q, t_q = torch.zeros(t_2x52.shape[0],device=device,dtype=INT32), torch.zeros((0,2,t_2x52.shape[2]), device=device,dtype=INT8)
    else:

        t_2x4                     = t_2x52[:,:,tm_q]
        tu_2x4, t_inv_inds = torch.unique(t_2x4,dim=0,return_inverse=True)
        tu_q, tuc_q               = k_actions(tu_2x4,t_m52[44:48], D_S2T)
        tc_q, t_q                 = tuc_q[t_inv_inds], tu_q[expand_unique(tuc_q,t_inv_inds)]

    
    
    t_pick = F.pad(t_pick,(0,i_pad_n))
    t_lay  = F.pad(t_lay,(0,i_pad_n))
    t_k    = F.pad(t_k,(0,i_pad_k))[:,:,t_pad_k]
    t_q    = F.pad(t_q,(0,i_pad_q))[:,:,t_pad_q]
    t_j    = F.pad(t_j,(0,i_pad_j))
    l_cnts = [tc_pick,tc_lay, tc_k,tc_q,tc_j]
    c_act = torch.sum(torch.stack(l_cnts), dim=0).to(INT32)
    # t_inds = torch.cat([torch.repeat_interleave(torch.arange(t_2x52.shape[0], device=device), cnt) for cnt in l_cnts], dim=0)
    t_inds = torch.cat([torch.repeat_interleave(torch.arange(t_2x52.shape[0], device=device), cnt) for cnt in l_cnts], dim=0)
    _, t_inds_sorted = torch.sort(t_inds)

    lc_t = [t_pick,t_lay, t_k,t_q,t_j]
    t_act = torch.cat(lc_t,dim=0)[t_inds_sorted]
    
    
    return t_act, c_act