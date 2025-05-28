import torch, sys, os
import torch.nn.functional as F
from Setup import setup
from Utils import get_d_mask, pad_helper
from FindMoves import find_moves
from ApplyMoves import apply_moves
from Imports import device, INT8, INT32, d_snw, d_scr
from RepeatedBlocks import repeatedblocks
from CleanPool import cleanpool 
from ExternalSampling import extsampling, sorttsid
from zstd_store import to_memory
import xgboost as xgb
import numpy as np 


def simulategamerun(seed, folder, x_alx, x_bob, i_ext, save_folder):
# def simulategamerun(seed, folder, x_alx, x_bob, i_ext,str_memory={}, adv_memory={}):
# def simulategamerun(seed, folder, nn_alx, nn_bob, i_ext,str_memory={}, adv_memory={}):

    # nn_alx.eval()
    # nn_bob.eval()
    # nn_type = nn_alx.layers[0].weight.dtype
    # seed = int(sys.argv[1])

    x_bob.set_param({"predictor": "cpu_predictor"})
    x_alx.set_param({"predictor": "cpu_predictor"})

    torch.manual_seed(seed)
    
    i_oth = (i_ext+1) % 2
    d_fls = {}
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    t_inf, t_sid, t_scr, t_dck, t_m52, t_mdd, clt_dck = setup()

    adv_t_nns  = {}
    str_t_nns  = {}
    str_t_sgm = {}

    for i_hnd in range(6):

        t_nnd = torch.zeros((1,52), device=device)
        if i_hnd > 0:
            
            i_nnd = 8*i_hnd+4
            t_nnd[0,clt_dck[:i_nnd]] = 1

        # print(i_hnd)
        # print(100*"**")
        torch.cuda.empty_cache()
        _, c_scr = torch.unique(t_sid[:,0],dim=0,return_counts=True)

        d_msk = get_d_mask(t_m52, t_dck[:8])
        t_dck = t_dck[8:]
        t_inf = F.pad(t_inf, (0, 8))[:,:,d_msk['pad']]
        t_inf[:,0, d_msk['nxt_a']] = 1
        t_inf[:,0, d_msk['nxt_b']] = 2
        i_pad_n, _       = pad_helper(d_msk, 'n') 
        i_pad_k, t_pad_k = pad_helper(d_msk, 'k')
        i_pad_q, t_pad_q = pad_helper(d_msk, 'q')
        i_pad_j, _       = pad_helper(d_msk, 'j')

        d_pad = {'i_pad_n': i_pad_n, 'i_pad_k':i_pad_k, 'i_pad_q':i_pad_q, 'i_pad_j':i_pad_j, 't_pad_k':t_pad_k, 't_pad_q':t_pad_q}
        t_snw = torch.zeros((t_inf.shape[0],len(d_snw)), device=device,dtype=INT8)
    
        # with open(os.path.join(folder, "seed.txt"), "w") as f:
        #     f.write(str(seed))

        for i_trn in range(4):
            for i_ply in range(2):
                
                i_cod = f'{i_hnd}_{i_trn}_{i_ply}'
                # print(i_cod)
                i_ind = 2 if i_ply == 0 else 1 
                i_sid = t_sid.shape[0]
                # print(i_sid)
                t_act, c_act  = find_moves(t_inf,i_ply, d_msk, d_pad) 

                t_inf         = torch.repeat_interleave(t_inf, c_act, dim=0)
                t_snw         = torch.repeat_interleave(t_snw, c_act, dim=0)

                t_inf, t_snw  = apply_moves(t_inf, t_act, t_snw, d_msk,  i_hnd, i_ply, i_trn)
                
                t_cl1         = repeatedblocks(t_sid[:,1], c_scr, c_act)
                t_edg         = repeatedblocks(torch.arange(t_sid.shape[0], device=device),c_scr,c_act)
                c_edg         = c_act.repeat_interleave(c_scr)
                c_scr         = torch.repeat_interleave(c_scr, c_act, dim=0)
                t_cl0         = torch.repeat_interleave(torch.arange(t_inf.shape[0],device=device), c_scr)
                t_sid         = torch.stack([t_cl0,t_cl1], dim=1)


                ## CONSTRUCT INFOSET I 
                
                t_tmp              = t_inf[t_sid[:,0]]

                t_xgl                 = t_tmp[:,1,:]-t_tmp[:,2,:]
                t_xgl[torch.logical_and(t_tmp[:,0,:]==3, t_xgl==0)] = 110
                t_xgl[torch.logical_and(t_tmp[:,0,:]==i_ply+1, t_xgl==0)] = 100
                t_xgl[torch.logical_and(t_nnd.squeeze(0)[t_m52] == 1, t_xgl==0)] = -127

                t_xgb          = torch.zeros((t_sid.shape[0],56), dtype=torch.int8, device=device)
                t_xgb[:,torch.cat([t_m52,torch.tensor([False,False,False,False], device=device)],dim=0)] = t_xgl
                t_xgb[:,52:] = t_scr[t_sid[:,1]]
                n_xgb = t_xgb.cpu().numpy()
                # n_xgb = xgb.DMatrix(n_xgb)
                
                # if i_hnd == 3:
                #     import pdb; pdb.set_trace()

                # t_nnl            = torch.zeros((t_sid.shape[0],4,52), dtype=nn_type, device=device)

                # t_nnl[:,:-1,t_m52] = t_inf[t_sid[:,0]].to(nn_type)
                # t_nnl[:,-1,:]      = t_nnd
                # t_nnl[:,0,:][t_nnl[:,0,:] == i_ind] = 0 
                # t_nnl = t_nnl.reshape(t_nnl.shape[0],-1)
                # t_nn = torch.cat([t_nnl,t_scr[t_sid[:,1]]], dim=1)  # t_nnr = t_scr[t_sid[:,1]]

                # del t_nnl 
                t_sum = torch.zeros(i_sid, dtype = torch.float32, device=device)  

                with torch.no_grad():

                    if i_ply == 0:
                        # t_adv = nn_alx(t_nn)
                        t_adv = torch.from_numpy(x_alx.predict(xgb.DMatrix(n_xgb))).to(device)
                    else:
                        t_adv = torch.from_numpy(x_bob.predict(xgb.DMatrix(n_xgb))).to(device)
                        # t_adv = nn_bob(t_nn)


                    t_adv = torch.clamp(t_adv, min=0)
                    t_sum.scatter_add_(0, t_edg, t_adv)
                    # t_sgm = t_adv/t_sum[t_edg]
                    t_sgm = torch.zeros(t_xgb.shape[0], dtype = t_sum.dtype, device=device)
                    # t_sgm = torch.zeros(t_nn.shape[0], dtype = t_sum.dtype, device=device)
                    t_msk = t_sum[t_edg] == 0 
                    t_szr = t_sum == 0
                    t_sgm[t_msk] = 1/torch.repeat_interleave(c_edg[t_szr], c_edg[t_szr]).to(t_sum.dtype)
                    t_sgm[~t_msk] = (t_adv[~t_msk]/t_sum[t_edg][~t_msk]).to(t_sum.dtype)
                

                

                # del t_nn 
                if i_ply == i_ext:

                    # save_folder = 
                    # str_t_nns[i_cod]  = t_nn.to(torch.int8)
                    str_t_nns[i_cod]  = n_xgb
                    str_t_sgm[i_cod]  = t_sgm
                    # if int(folder.split('_')[1][3:]) == 9 and i_hnd == 1 and i_trn == 1: import pdb; pdb.set_trace()
                    

                    # t_exs = extsampling(t_edg, c_edg, t_sgm)
                    # t_idx = t_sid[t_exs,0].unique_consecutive()
                    # t_inf = t_inf[t_idx]
                    # t_snw = t_snw[t_idx]
                    # t_sid = t_sid[t_exs]
                    # t_edg = t_edg[t_exs]
                    # t_sid[:,0] = sorttsid(t_sid[:,0])
                    # _, c_scr = torch.unique(t_sid[:,0],dim=0,return_counts=True)
                else:

                    adv_t_nns[i_cod] = n_xgb
                    # adv_t_nns[i_cod] = t_nn.to(torch.int8)
                    d_fls[f'i_nns_{i_cod}']  = i_sid
                    d_fls[f't_sgm_{i_cod}'] = t_sgm
                    # if int(folder.split('_')[1][3:]) == 9 and i_hnd == 1 and i_trn == 1: import pdb; pdb.set_trace()


                d_fls[f't_edg_{i_cod}'] = t_edg
                
          
        if i_hnd ==  5:  t_inf, t_snw = cleanpool(t_inf, t_snw, d_msk)

        t_inf        = t_inf[:,0,:]
        t_snw[:,d_snw['pts_dlt']]   = t_snw[:,d_snw['a_pts']] + t_snw[:,d_snw['a_sur']] - t_snw[:,d_snw['b_pts']] - t_snw[:,d_snw['b_sur']]
        t_snw        = t_snw[:,:4]
        t_snw[:,-1]  = 0    
        t_inf, t_lnk = torch.unique(t_inf,dim=0, sorted=False, return_inverse=True)
        t_snw, t_wid = torch.unique(t_snw,dim=0, sorted=False, return_inverse=True) 
        t_prs        = torch.stack([t_sid[:,1], t_wid[t_sid[:,0]]],dim=1)
        t_sid[:,0]   = t_lnk[t_sid[:,0]]

        t_prs,t_pid  = torch.unique(t_prs,dim=0,sorted=False,return_inverse=True)
        t_scr        = t_scr[t_prs[:,0]]+t_snw[t_prs[:,1]]

        t_mal = t_scr[:,d_scr['a_clb']]>=7
        t_mbb = t_scr[:,d_scr['b_clb']]>=7
        t_scr[:,d_scr['max_clb']][t_mal]= 1
        t_scr[:,d_scr['max_clb']][t_mbb]= 2
        t_mcl = t_scr[:,d_scr['max_clb']]>0
        t_scr[:,0:2].masked_fill_(t_mcl.unsqueeze(-1), 0)


        t_scr,t_fid  = torch.unique(t_scr, dim=0,sorted=False,return_inverse=True)
        t_sid[:,1]   = t_fid[t_pid]
        t_sid, t_lnk = torch.unique(t_sid,dim=0,return_inverse=True)
        # if saving:  to_zstd([t_lnk], ['t_lnk'], folder, i_hnd) 
        d_fls[f't_lnk_{i_hnd}'] = t_lnk
        
        t_c52        = t_m52.clone()
        t_m52[t_c52] = torch.any(t_inf>0, dim=0)
        t_inf        = t_inf[:,t_m52[t_c52]]
        t_ndd        = torch.logical_and(t_c52,~t_m52)
        t_mdd[t_ndd] = True
        t_inf        = t_inf.unsqueeze(1)
        
        if t_inf.shape[2] == 0:
            t_inf = torch.empty((t_inf.shape[0], 3, 0), device='cuda:0', dtype=INT8)
        else:
            t_inf = F.pad(t_inf, (0, 0, 0, 2))

    d_fls['t_scr_6'], d_fls['t_sid_6'] = t_scr, t_sid
    
    # s_name  = f'../STRG/CFR_{i_cfr}/EXT_{i_ext}/TRV_{i_trv}/'
    
    str_keys = [
                f'{i_hnd}_{i_trn}_{i_ext}'
                for i_hnd in range(6)
                for i_trn in range(4)
                ]

    adv_keys = [
                f'{i_hnd}_{i_trn}_{i_oth}'
                for i_hnd in range(6)
                for i_trn in range(4)
                ]
    
    # if type(adv_t_nns[f'0_0_{i_oth}']).__module__.split('.')[0] == 'numpy':
        
    # str_memory[folder+'_nn']  = np.concatenate([str_t_nns[id] for id in str_keys] ,0)
    # xx = np.concatenate([str_t_nns[id] for id in str_keys] ,0)
    # yy = torch.cat([str_t_sgm[id] for id in str_keys], dim=0).to(torch.float16)
    
    
    # adv_memory[folder+'_nn']  = np.concatenate([adv_t_nns[id] for id in adv_keys], axis=0)

    to_memory(torch.from_numpy(np.concatenate([adv_t_nns[id] for id in adv_keys], axis=0)),  f"{save_folder}/ADVT/nns/{folder}.pt.zst")
    to_memory(torch.from_numpy(np.concatenate([str_t_nns[id] for id in str_keys] ,0)),  f"{save_folder}/STRG/nns/{folder}.pt.zst")
    to_memory(torch.cat([str_t_sgm[id] for id in str_keys], dim=0).cpu(),  f"{save_folder}/STRG/sgm/{folder}.pt.zst")

    # to_memory(torch.cat([str_t_sgm[id].to('cpu') for id in str_keys],0),  folder+'_Str_t_sgm.zstd', device='disk', memory_store=str_memory, compress = False)


    # torch.save(torch.from_numpy(np.concatenate([str_t_nns[id] for id in str_keys] ,0)), f"../STRG/nns/{folder}_nn.pt")
    # torch.save(torch.cat([str_t_sgm[id] for id in str_keys], dim=0).to(torch.float16),f"../STRG/sgm/{folder}_sgm.pt")
    # str_memory[folder+'_sgm'] = torch.cat([str_t_sgm[id] for id in str_keys], dim=0).to('cpu').numpy()


    #     to_memory(torch.cat([str_t_nns[id] for id in str_keys],0).to('cpu').numpy(),  folder+'_Str_t_nn.zstd',  device='cpu' ,  memory_store=str_memory, compress = False)
    #     to_memory(np.concatenate([adv_t_nns[id] for id in adv_keys], axis=0),  folder+'_Adv_t_nn.zstd',  device='cpu' ,  memory_store=adv_memory, compress = False)
    #     to_memory(np.concatenate([str_t_sgm[id] for id in str_keys], axis=0),  folder+'_Str_t_sgm.zstd', device='cpu' ,  memory_store=str_memory, compress = False)
    
    # else:
    #     to_memory(torch.cat([str_t_nns[id] for id in str_keys],0),  folder+'_Str_t_nn.zstd', device='cpu' , memory_store=str_memory, compress = False)
    #     to_memory(torch.cat([adv_t_nns[id] for id in adv_keys],0),  folder+'_Adv_t_nn.zstd', device='cpu' , memory_store=adv_memory, compress = False)
    #     to_memory(torch.cat([str_t_sgm[id].to('cpu') for id in str_keys],0),  folder+'_Str_t_sgm.zstd', device='cpu', memory_store=str_memory, compress = False)

    # import pdb; pdb.set_trace()
    # print(adv_keys)
    # import pdb; pdb.set_trace()



    # to_zstd([torch.cat(str_t_sgm,0)], ['t_sgm'] , folder+f'StratMemory',to_memory=True)
    # to_zstd([torch.cat(adv_t_nn,0)], ['t_nn'] , folder+f'AdvanMemory',to_memory=True)
    # to_zstd([torch.cat(str_t_nn,0)], ['t_nn'] , folder+f'StratMemory',to_memory=True)
    # if saving: to_zstd([t_inf, t_sid], ['t_inf', 't_sid'] , 
    #     save_folder, i_cod)

    return d_fls
    # return str_memory, adv_memory, d_fls


    