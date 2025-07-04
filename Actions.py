import torch 
# from Utils import D_S2T

TYPE = torch.int8
TYPE2 = torch.int32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\

def f(t,tm):

    t2 = torch.zeros(4, device = device, dtype=torch.int8)
    t2[tm] = t
    s_t2 = f'{t2[0]}{t2[1]}{t2[2]}{t2[3]}'
    if '2' in s_t2:
        return s_t2+'0'
    else:
        return s_t2+'2'


def n_actions(t_2x40, t_2764x40, t_40x2764, t_tuples):

    t_2x40_f16 = t_2x40.to(torch.float16)
    m = t_2x40_f16.shape[-1]

    ## STEP 1:

    t_tmp =  t_tuples-torch.matmul(t_2x40_f16,t_40x2764)    
    t_inds = torch.nonzero((t_tmp[:, 0, :] == 0) & (t_tmp[:, 1, :] == 0))
    tc_pick = t_inds[:, 0].bincount(minlength=t_2x40_f16.shape[0]).to(TYPE2)
    tc_pick_p = tc_pick>0
    t_tmp  = tc_pick[tc_pick_p]
    t_pick = torch.logical_and(t_2x40_f16[tc_pick_p].repeat_interleave(t_tmp,dim=0),t_2764x40[t_inds[:,1]].unsqueeze(1).repeat_interleave(2, dim=1)).to(TYPE) 
    # import pdb; pdb.set_trace()
    ## STEP 2:

    # Construct t_hand
    
    t_hand = torch.zeros((t_tmp.shape[0], m), device=device, dtype=TYPE)
    t_hand_inds = torch.repeat_interleave(torch.arange(t_tmp.shape[0], device=device), t_tmp).view(-1,1).expand(-1,m)
    t_src = t_pick[:,0,:]
    t_hand.scatter_add_(0,t_hand_inds,t_src)
    
    # # Construct t_lay 

    t_tmp = t_2x40[:,0,:].clone()
    t_tmp[tc_pick_p,:] = torch.relu(t_2x40[tc_pick_p,0,:]-t_hand)
    t_inds = torch.nonzero(t_tmp == 1)
    tc_lay = t_inds[:,0].bincount(minlength=t_tmp.shape[0]).to(TYPE2)
    t_lay = torch.zeros((t_inds.shape[0],2,m), device=device, dtype=TYPE)
    t_lay[torch.arange(t_lay.shape[0], device=device),0,t_inds[:, 1]] = 1
    import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    return t_pick, tc_pick,t_lay, tc_lay


# j_actions(t_2x44 = t_2x52[:,:,t_cur_n+t_cur_j], t_cur_j=t_cur_j)
def j_actions(t_2x44,i_cur_n):

    ##TODO: find the index of the first j -> modify utils.py
    t_tmp = t_2x44[:,0,i_cur_n:]
    t_j_inds = torch.nonzero(t_tmp)
    tc_j = t_j_inds[:,0].bincount(minlength=t_2x44.shape[0]).to(TYPE2)
    t_j = torch.zeros((t_j_inds.shape[0], 2, t_2x44.shape[2]), device=device, dtype=TYPE)
    t_arange = torch.arange(t_j_inds.shape[0],device=device)
    t_j[t_arange, 0, i_cur_n+t_j_inds[:,1]] = 1
    t_j[t_arange,1,:] = torch.repeat_interleave(t_2x44[:,1,:], tc_j, dim=0)
    return t_j, tc_j



def k_actions(t_2x4,tm, D_S2T):

    t_2x4[:, 1, :][t_2x4[:, 1, :] == 1] = 2
    t_tmp = t_2x4.sum(dim=1).to(TYPE)
    d_tmp = {i:D_S2T[f(t_tmp[i,:],tm)] for i in range(t_2x4.shape[0])}
    tc_k  = torch.tensor([d_tmp[i].shape[0] for i in range(t_2x4.shape[0])], device=device, dtype=TYPE2)
    t_k = torch.cat([d_tmp[i] for i in range(t_2x4.shape[0])], dim = 0)
    return t_k[:,:,tm], tc_k




