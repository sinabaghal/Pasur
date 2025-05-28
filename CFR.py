import os, torch
from ZipTools import to_zstd, from_zstd
from RepeatedBlocks import repeatedblocks

seed = 20
d_fls = from_zstd(seed)
t_dck, t_sid, t_scr, t_m52 = d_fls['t_dck_0'], d_fls['t_sid_6'], d_fls['t_scr_6'], d_fls[f't_m52_6']

t_fsc = 7*(t_scr[:,-1] % 2)+t_scr[:,-2]
t_val = t_fsc[t_sid[:,1]].to(torch.float32)
for i_hnd in reversed(range(6)):

    t_scr, t_m52, t_lnk  = d_fls[f't_scr_{i_hnd}'],  d_fls[f't_m52_{i_hnd}'], d_fls[f't_lnk_{i_hnd}'] 
    t_val = t_val[t_lnk]
    
    for i_trn in reversed(range(4)):
        for i_ply in reversed(range(2)):
            
            i_cod    = f'{i_hnd}_{i_trn}_{i_ply}'
            print(i_cod)
            t_edg, t_sid, t_inf    = d_fls[f't_edg_{i_cod}'], d_fls[f't_sid_{i_cod}'], d_fls[f't_inf_{i_cod}']
            import pdb; pdb.set_trace()
    
    t_sid =   d_fls[f't_sid_{i_hnd}']
import pdb; pdb.set_trace()