import torch 
from Imports import device
def extsampling(t_edg, c_edg, t_sgm):

    t_inv         = torch.argsort(t_edg,stable=True)
    t_idx         = torch.empty_like(t_inv, device=device)
    t_idx[t_inv]  = torch.arange(len(t_inv),device=device)
    i_max         = c_edg.max()
    t_msk         = torch.arange(i_max, device=device).unsqueeze(0) < c_edg.unsqueeze(1)
    t_mtx         = torch.zeros_like(t_msk, dtype=t_sgm.dtype)
    t_mtx[t_msk]  = t_sgm[t_inv]
    t_smp         = torch.multinomial(t_mtx, num_samples=1).squeeze(1) 
    t_gps         = torch.cat([torch.tensor([0], device=device), c_edg.cumsum(0)[:-1]]) # group starts 
    
    t_res         = torch.zeros_like(t_sgm, dtype=torch.bool, device=device)
    t_res[t_gps + t_smp] = True 
    t_res         = t_res[t_idx]

    return t_res



def sorttsid(t_col):

    t_new = torch.ones_like(t_col, dtype=torch.bool)
    t_new[1:] = t_col[1:] != t_col[:-1]
    t_out = torch.cumsum(t_new, dim=0) - 1

    return t_out