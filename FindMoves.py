from Actions import n_actions, j_actions, k_actions
import torch.nn.functional as F
from Utils import inverseunique
from Imports import device, INT8, INT32, t_40x2764, t_2764x40, t_tpl, D_S2T
import torch 

def find_moves(t_gme, i_ply, d_msk, d_pad):

    t_inn  = d_msk['cur_n']
    t_inj  = d_msk['cur_j']
    t_inq  = d_msk['cur_q']
    t_ink  = d_msk['cur_k']
    t_inp  = d_msk['cur_52']
    i_pdn  = d_pad['i_pdn']
    i_pdk  = d_pad['i_pdk']
    i_pdq  = d_pad['i_pdq']
    i_pdj  = d_pad['i_pdj']
    t_pdk  = d_pad['t_pdk']
    t_pdq  = d_pad['t_pdq']

   

    t_2x52 = torch.zeros((t_gme.shape[0],2,t_gme.shape[2]), dtype=INT8, device=device) 
    tm_plyr = t_gme[:,0,:] == i_ply+1
    tm_pool = t_gme[:,0,:] == 3
    t_2x52[:,0,:][tm_plyr] = 1
    t_2x52[:,1,:][tm_pool] = 1
    # import pdb; pdb.set_trace()


    ### Construct numerical actions 

    t_2x40 = t_2x52[:,:,t_inn]
    
    tu_2x40, t_inx = torch.unique(t_2x40,dim=0, sorted = False, return_inverse=True)
    # tu_2x40, t_inx = torch.unique(t_2x40,dim=0, return_inverse=True)
    tu_pck, cu_pck,tu_lay, cu_lay = \
            n_actions(tu_2x40, t_2764x40[:,t_inp[:40]], t_40x2764[t_inp[:40],:],t_tpl) 

    c_pck, c_lay = cu_pck[t_inx], cu_lay[t_inx]
    t_pck, t_lay = tu_pck[inverseunique(cu_pck,t_inx)], tu_lay[inverseunique(cu_lay,t_inx)]
    ### Construct jack actions
    
    t_j, c_j =  j_actions(t_2x44 = t_2x52[:,:,t_inn+t_inj], i_pdn = t_inn.sum())
    # import pdb; pdb.set_trace()
    ### Construct king actions 
    
    if t_inp[48:52].sum() == 0 :
        c_k, t_k = torch.zeros(t_2x52.shape[0],device=device,dtype=INT32), torch.zeros((0,2,t_2x52.shape[2]), device=device,dtype=INT8)
    else:
        t_2x4 = t_2x52[:,:,t_ink]
        tu_2x4, t_inx = torch.unique(t_2x4,dim=0, sorted = False, return_inverse=True)
        # tu_2x4, t_inx = torch.unique(t_2x4,dim=0,return_inverse=True)
        tu_k, tu_k_cnt            = k_actions(tu_2x4,t_inp[48:52], D_S2T)
        c_k, t_k                 = tu_k_cnt[t_inx], tu_k[inverseunique(tu_k_cnt,t_inx)]
    ### Construct queen actions 

    if t_inp[44:48].sum() == 0:
        c_q, t_q = torch.zeros(t_2x52.shape[0],device=device,dtype=INT32), torch.zeros((0,2,t_2x52.shape[2]), device=device,dtype=INT8)
    else:

        t_2x4                     = t_2x52[:,:,t_inq]
        tu_2x4, t_inx = torch.unique(t_2x4, dim=0, sorted=False, return_inverse=True)
        # tu_2x4, t_inx = torch.unique(t_2x4,dim=0,return_inverse=True)
        tu_q, tuc_q               = k_actions(tu_2x4,t_inp[44:48], D_S2T)
        c_q, t_q                 = tuc_q[t_inx], tu_q[inverseunique(tuc_q,t_inx)]

    
    
    t_pck = F.pad(t_pck,(0,i_pdn))
    t_lay  = F.pad(t_lay,(0,i_pdn))
    t_k    = F.pad(t_k,(0,i_pdk))[:,:,t_pdk]
    t_q    = F.pad(t_q,(0,i_pdq))[:,:,t_pdq]
    t_j    = F.pad(t_j,(0,i_pdj))
    l_cnts = [c_pck,c_lay, c_k,c_q,c_j]
    # import pdb; pdb.set_trace()
    t_brf = torch.sum(torch.stack(l_cnts), dim=0)
    # import pdb; pdb.set_trace()
    # c_act = torch.sum(torch.stack(l_cnts), dim=0).to(INT32)
    # t_inds = torch.cat([torch.repeat_interleave(torch.arange(t_2x52.shape[0], device=device), cnt) for cnt in l_cnts], dim=0)
    t_inds = torch.cat([torch.repeat_interleave(torch.arange(t_2x52.shape[0], device=device), cnt) for cnt in l_cnts], dim=0)
    _, t_inds_sorted = torch.sort(t_inds)

    lc_t = [t_pck,t_lay, t_k,t_q,t_j]
    t_act = torch.cat(lc_t)[t_inds_sorted]
    # t_act = torch.cat(lc_t,dim=0)[t_inds_sorted]
    # assert torch.all(t_act == torch.cat(lc_t)[t_inds_sorted])
    # import pdb; pdb.set_trace()
    
    
    return t_act, t_brf