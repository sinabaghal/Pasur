from Imports import device, INT8, INT32, d_snw, d_scr
import torch 

def strategyeval(t_inf,t_scr,t_sid,t_edg,V_pnn):

    i_vk  = t_sid.shape[0]
    t_sum = torch.zeros(i_vk, device = device)
    t_nnl = t_inf[t_sid[:,0]]
    t_nnr = t_scr[t_sid[:,1]]
    t_nn  = torch.stack([t_nnl,t_nnr],dim=1)
    t_reg = V_pnn(t_nn)
    t_sum.scatter_add_(0,t_edg,t_reg)
    t_sig = t_reg/t_sum[t_edg]

    return t_sig, t_reg 





