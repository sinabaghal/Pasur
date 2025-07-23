import torch 
# from Utils import D_S2T
from Imports import device
TYPE = torch.int8
TYPE2 = torch.int32
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\

def f(t,tm):

    t2 = torch.zeros(4, device = device, dtype=torch.int8)
    t2[tm] = t
    s_t2 = f'{t2[0]}{t2[1]}{t2[2]}{t2[3]}'
    if '2' in s_t2:
        return s_t2+'0'
    else:
        return s_t2+'2'


def n_actions(t_2x40, t_2764x40, t_40x2764, t_tpl):

    t_2x40_f16 = t_2x40.to(torch.float16)
    m = t_2x40_f16.shape[-1]

    ## STEP 1:

    # t_tmp  = t_tpl-torch.matmul(t_2x40_f16,t_40x2764)    
    t_inx = torch.nonzero(((t_tpl-torch.matmul(t_2x40_f16,t_40x2764)) == 0).all(dim=1))
    # t_inx = torch.nonzero(((t_tpl-torch.matmul(t_2x40_f16,t_40x2764))[:, :2, :] == 0).all(dim=1))
    # t_inds2 = torch.nonzero((t_tmp[:, 0, :] == 0) & (t_tmp[:, 1, :] == 0))
    # assert torch.all(t_inx == t_inds2) 
    # torch.all(torch.nonzero((t_tmp[:, :2, :] == 0).all(dim=1)) == t_inx)
    # torch.nonzero(t_tmp[:, :, :] == 0)
    # import pdb; pdb.set_trace()
    # c_pck  = t_inx[:, 0].bincount(minlength=t_2x40_f16.shape[0]).to(torch.int8)
    c_pck  = t_inx[:, 0].bincount(minlength=t_2x40_f16.shape[0]).to(TYPE2)
    m_pck  = c_pck>0
    t_tmp  = c_pck[m_pck]
    # t_pck  = torch.logical_and(t_2x40_f16[m_pck].repeat_interleave(t_tmp,dim=0), t_2764x40[t_inx[:,1]].unsqueeze(1).repeat_interleave(2, dim=1)).to(TYPE) 
    t_pck  = torch.logical_and(t_2x40_f16.repeat_interleave(c_pck,dim=0), t_2764x40[t_inx[:,1]].unsqueeze(1).repeat_interleave(2, dim=1)).to(TYPE) 
    # torch.all(t_pck == t_pck2)
    # import pdb; pdb.set_trace()
    ## STEP 2:

    # Construct t_hnd
    
    t_hnd = torch.zeros((t_tmp.shape[0], m), device=device, dtype=TYPE)
    t_hand_inds = torch.repeat_interleave(torch.arange(t_tmp.shape[0], device=device), t_tmp).view(-1,1).expand(-1,m)
    t_src = t_pck[:,0,:]
    # import pdb; pdb.set_trace()
    t_hnd.scatter_add_(0,t_hand_inds,t_src)
    
    # # Construct t_lay 

    t_cln = t_2x40[:,0,:].clone()
    t_cln[m_pck,:] = torch.relu(t_2x40[m_pck,0,:]-t_hnd)
    t_inx = torch.nonzero(t_cln==1)
    # c_lay  = t_inx[:,0].bincount(minlength=t_tmp.shape[0]).to(torch.int8)
    c_lay  = t_inx[:,0].bincount(minlength=t_2x40.shape[0]).to(TYPE2)
    # c_lay  = t_inx[:,0].bincount(minlength=t_tmp.shape[0]).to(TYPE2)
    t_lay  = torch.zeros((t_inx.shape[0],2,m), device=device, dtype=TYPE)
    t_lay[torch.arange(t_lay.shape[0], device=device),0,t_inx[:, 1]] = 1
    # import pdb; pdb.set_trace()
    return t_pck, c_pck,t_lay, c_lay


# j_actions(t_2x44 = t_2x52[:,:,t_cur_n+t_cur_j], t_cur_j=t_cur_j)
def j_actions(t_2x44,i_pdn):

    ##TODO: find the index of the first j -> modify utils.py
    # t_tmp = t_2x44[:,0,i_pdn:]
    # t_inx = torch.nonzero(t_tmp)
    t_inx = torch.nonzero(t_2x44[:,0,i_pdn:])
    i_cnt = t_inx.shape[0]

    # c_jck = t_inx[:,0].bincount(minlength=t_2x44.shape[0]).to(torch.int8)
    c_jck = t_inx[:,0].bincount(minlength=t_2x44.shape[0]).to(TYPE2)
    t_jck = torch.zeros((i_cnt, 2, t_2x44.shape[2]), device=device, dtype=TYPE)
    # t_jck = torch.zeros((t_inx.shape[0], 2, t_2x44.shape[2]), device=device, dtype=TYPE)
    
    # t_arg = torch.arange(t_inx.shape[0],device=device)
    t_arg = torch.arange(i_cnt,device=device)
    t_jck[t_arg, 0, i_pdn+t_inx[:,1]] = 1
    t_jck[t_arg,1,:] = torch.repeat_interleave(t_2x44[:,1,:], c_jck, dim=0)
    return t_jck, c_jck



def k_actions(t_2x4,tm, D_S2T):

    t_2x4[:, 1, :][t_2x4[:, 1, :] == 1] = 2
    t_tmp = t_2x4.sum(dim=1).to(TYPE)
    d_tmp = {i:D_S2T[f(t_tmp[i,:],tm)] for i in range(t_2x4.shape[0])}
    # c_k   = torch.tensor([d_tmp[i].shape[0] for i in range(t_2x4.shape[0])], device=device, dtype=torch.int8)
    c_k   = torch.tensor([d_tmp[i].shape[0] for i in range(t_2x4.shape[0])], device=device, dtype=TYPE2)
    t_k   = torch.cat([d_tmp[i] for i in range(t_2x4.shape[0])], dim = 0)
    return t_k[:,:,tm], c_k



def k_act(t_2x4,tm,D_S2T):

    t_2x4[:, 1, :][t_2x4[:, 1, :] == 1] = 2
    t_tmp = t_2x4.sum(dim=1).to(TYPE)
    d_tmp = {i:D_S2T[f(t_tmp[i,:],tm)] for i in range(t_2x4.shape[0])}
    # c_k   = torch.tensor([d_tmp[i].shape[0] for i in range(t_2x4.shape[0])], device=device, dtype=torch.int8)
    c_k   = torch.tensor([d_tmp[i].shape[0] for i in range(t_2x4.shape[0])], device=device, dtype=TYPE2)
    t_k   = torch.cat([d_tmp[i] for i in range(t_2x4.shape[0])], dim = 0)
    import pdb; pdb.set_trace()
    return t_k[:,:,tm], c_k

