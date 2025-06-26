from Imports import  INT8, d_snw

def cleanpool(t_inf, t_snw, d_msk):
    # import pdb; pdb.set_trace()
    tm_p     = d_msk['cur_p']
    tm_s     = d_msk['cur_s']
    tm_c     = d_msk['cur_c']
    
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

    # import pdb; pdb.set_trace()
    # t_snw[:,d_snw['a_pts']].unique()
    t_snw[tm_la,d_snw['a_pts']] += (t_pla[:,tm_p]*tm_s).sum(dim=1).to(INT8)
    t_snw[tm_lb,d_snw['b_pts']] += (t_plb[:,tm_p]*tm_s).sum(dim=1).to(INT8)

    # import pdb; pdb.set_tracet_snw()
    t_inf[:,0,:] = 0
    return t_inf, t_snw