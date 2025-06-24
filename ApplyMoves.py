import torch 
from Imports import INT8, d_snw 


def apply_moves(t_inf, t_act, t_snw, d_msk, i_hnd, i_ply, i_trn):

    tm_p     = d_msk['cur_p']
    tm_s     = d_msk['cur_s']
    tm_c     = d_msk['cur_c']
    tm_j     = d_msk['cur_j']
    t_mpk             =  torch.any(t_act[:,1,:],dim=1) 
    tm_pick_notj      =  torch.any(t_act[:,0,tm_j],dim=1) == 0
    t_inf[~t_mpk,0,:] +=(2-i_ply)*t_act[~t_mpk,0,:] ## add to pool
    t_inf[t_mpk,0,:]  -= (1+i_ply)*t_act[t_mpk,0,:] ## remove to pool
    t_inf[t_mpk,0,:] -= 3*t_act[t_mpk,1,:]   ## remove from pool
    t_inf[:,i_ply+1,:] += (i_trn+1)*t_act[:,0,:]+10*(i_trn+1)*t_act[:,1,:] # v*card held by player + 10*v*cards picked from the pool  

    ## apply_moves score tensor
    plyr = 'a' if i_ply ==0 else 'b'
    clubs_cntr, pt_cntr, surs_cntr = d_snw[f'{plyr}_clb'], d_snw[f'{plyr}_pts'],d_snw[f'{plyr}_sur']
    # clubs_cntr, pt_cntr = d_snw[f'{plyr}_clb'], d_snw[f'{plyr}_pts']

    
    t_clt                    = t_act[t_mpk, 1, :]+t_act[t_mpk, 0, :]
    t_snw[t_mpk,clubs_cntr] += t_clt[:,tm_c].sum(dim=1).to(INT8)
    t_snw[t_mpk,pt_cntr]    += (t_clt[:,tm_p]*tm_s).sum(dim=1).to(INT8)
    
    t_snw[t_mpk,d_snw['lst_pck']] = i_ply+1

    if i_hnd < 5:

        # if i_hnd ==1 and i_trn==0 and i_ply ==1 :
        #     import pdb; pdb.set_trace()
        t_sur = torch.any(t_inf[:, 0, :]==3,dim=1) == 0 
        t_msr = torch.logical_and(torch.logical_and(t_mpk, tm_pick_notj), t_sur)
        # if torch.any(t_msr):
        #     import pdb; pdb.set_trace()
        t_snw[t_msr,surs_cntr] += 5 ### PUT BACK
        # import pdb; pdb.set_trace()
        # if f'{i_hnd}_{i_trn}_{i_ply}' == '3_3_0':
        #     import pdb; pdb.set_trace()
        # if t_msr.sum()>0: import pdb; pdb.set_trace()
    
    assert t_inf.dtype == INT8
    return t_inf, t_snw