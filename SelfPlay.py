import torch, sys, os, math
import torch.nn.functional as F
from FindMoves import find_moves
from ApplyMoves import apply_moves
from Imports import device, INT8, INT32, d_snw, d_scr
from CleanPool import cleanpool 
from NeuralNet import SNN, init_weights_zero
import torch.nn as nn
from Utils import pad_helper
from tqdm import trange
import xgboost as xgb
from tqdm import trange, tqdm
import numpy as np 
from zstd_store import load_tensor 
import matplotlib.pyplot as plt



def selfplay(seed=10, N=4000, to_latex = False, x_alx = None,x_bob=None):


    t_m52 = torch.tensor([True for i in range(52)], device=device)
    find_dm = lambda lm :  torch.tensor([idx in lm for idx in torch.arange(52, device=device)], dtype = bool, device=device)
    jack_set = {40, 41, 42, 43}

    d_msk = {}
    d_msk['cur_k'] = find_dm(torch.arange(48,52, device=device))
    d_msk['cur_q'] = find_dm(torch.arange(44,48, device=device))
    d_msk['cur_j'] = find_dm(torch.arange(40,44, device=device))
    d_msk['cur_n'] = find_dm(torch.arange(40, device=device))
    d_msk['cur_c'] = find_dm(4*torch.arange(13, device=device))
    d_msk['cur_p'] = find_dm(torch.tensor([0,1,2,3,4,37,40,41,42,43], device=device))
    d_msk['cur_s'] = torch.tensor([1,1,1,1,2,3,1,1,1,1],dtype = INT8, device=device)
    d_msk['cur_52']= t_m52

    i_pad_n, _       = pad_helper(d_msk, 'n') 
    i_pad_k, t_pad_k = pad_helper(d_msk, 'k')
    i_pad_q, t_pad_q = pad_helper(d_msk, 'q')
    i_pad_j, _       = pad_helper(d_msk, 'j')
    d_pad = {'i_pad_n': i_pad_n, 'i_pad_k':i_pad_k, 'i_pad_q':i_pad_q, 'i_pad_j':i_pad_j, 't_pad_k':t_pad_k, 't_pad_q':t_pad_q}



    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    t_ltx   = {}

    # while True:

    #     noise = torch.rand(N * 2, 52, device=device)
    #     decks = torch.argsort(noise, dim=1)

    #     first_four = decks[:, :4]
    #     has_jack = (first_four.unsqueeze(-1) == torch.tensor(list(jack_set), device=device)).any(dim=(1, 2))

    #     valid_decks = decks[~has_jack]

    #     if valid_decks.size(0) >= N:
    #         break  
        
    t_dck = torch.tensor([47, 40, 45, 16, 23, 20,  1,  4, 21, 17, 36, 29, 18, 32, 49,  9, 34, 24,
        37, 41, 48, 44, 27,  8, 51,  2,  6,  0, 11, 10, 26, 33, 12, 42, 15, 13,
            3, 50, 39, 35, 43, 28,  7, 46, 30, 22, 19,  5, 25, 38, 31, 14],
        device='cuda:0')
    
    # t_dck.repeat(N,1)
    # import pdb; pdb.set_trace()
    t_dck  = t_dck.repeat(N,1)
    # t_dck = t_dck[111:113,:]
    clt_dck = t_dck.clone()
    t_inf   = torch.zeros((t_dck.shape[0],3,52), dtype=INT8, device=device)
    t_idx = torch.arange(t_inf.shape[0], device=t_dck.device).unsqueeze(1).expand(-1, 4)
    t_inf[t_idx,0,t_dck[:,:4]] = 3

    t_dck = t_dck[:,4:]
    t_idx = torch.arange(t_inf.shape[0], device=t_dck.device).unsqueeze(1).expand(-1, 4)
    t_nnd = torch.zeros((t_inf.shape[0],52), dtype =torch.float32, device=device)
    t_scr = torch.zeros((t_inf.shape[0],len(d_scr)), device=device,dtype=INT8)

    for i_hnd in range(6):

        t_snw     = torch.zeros((t_inf.shape[0],len(d_snw)), device=device,dtype=INT8)

        if i_hnd > 0: 
            t_nnd[:,clt_dck[:8*i_hnd+4]] = 1
        # torch.cuda.empty_cache()
        t_inf[t_idx,0, t_dck[:,:4]]  = 1
        t_inf[t_idx,0, t_dck[:,4:8]] = 2
        t_dck                        = t_dck[:,8:]
        
        if to_latex: t_ltx[f't_scr_{i_hnd}'] = t_scr.squeeze(0).clone()

        for i_trn in range(4):
            for i_ply in range(2):
                
                i_cod = f'{i_hnd}_{i_trn}_{i_ply}'
                i_ind = 2 if i_ply == 0 else 1 

                t_act, c_act  = find_moves(t_inf,i_ply, d_msk, d_pad) 
                if to_latex:
                    t_ltx['t_dck_'+i_cod]    = t_inf[:,0,:].clone()
            

                # mask = torch.zeros(t_inf.shape[0], device=device, dtype=torch.bool)
                # mask[111] = True
                t_inf          = torch.repeat_interleave(t_inf, c_act, dim=0)
                t_snw          = torch.repeat_interleave(t_snw, c_act, dim=0)
                # t_tst          = torch.repeat_interleave(mask , c_act, dim=0)

                # if f'{i_hnd}_{i_trn}_{i_ply}' == '3_3_0':
                #     import pdb; pdb.set_trace()
                    # 119 

                t_inf, t_snw   = apply_moves(t_inf, t_act, t_snw, d_msk,  i_hnd, i_ply, i_trn)



                t_xgl                 = t_inf[:,1,:]-t_inf[:,2,:]
                t_xgl[torch.logical_and(t_inf[:,0,:]==3, t_xgl==0)] = 110
                t_xgl[torch.logical_and(t_inf[:,0,:]==i_ply+1, t_xgl==0)] = 100
                t_xgl[torch.logical_and(torch.repeat_interleave(t_nnd, c_act,dim=0)==1, t_xgl==0)] = -127
                t_xgb          = torch.zeros((t_inf.shape[0],56), dtype=torch.int8, device=device)
                t_xgb[:,torch.cat([t_m52,torch.tensor([False,False,False,False], device=device)],dim=0)] = t_xgl

                t_xgb[:,52:] = torch.repeat_interleave(t_scr, c_act, dim=0)
                n_xgb = t_xgb.cpu().numpy()


                # t_nnl          = torch.zeros((t_inf.shape[0],4,52), dtype=torch.float32, device=device)
                # t_nnl[:,:-1,:] = t_inf.to(torch.float32)
                # t_nnl[:,-1,:]  = torch.repeat_interleave(t_nnd, c_act,dim=0)
                # t_nnl[:,0,:][t_nnl[:,0,:] == i_ind] = 0 
                # t_nnl          = t_nnl.reshape(t_nnl.shape[0],-1)
                # t_nn           = torch.cat([t_nnl,torch.repeat_interleave(t_scr, c_act, dim=0)], dim=1)  # t_nnr = t_snw[t_sid[:,1]]

                x_ply            =  x_alx if i_ply == 0 else x_bob
                # snn            = snn_alx if i_ply == 0 else snn_bob
                t_sgm = torch.from_numpy(x_ply.predict(xgb.DMatrix(n_xgb))).to(device)

                # import pdb; pdb.set_trace()
                t_sgm  = torch.min(torch.maximum(t_sgm, torch.tensor(0.001, dtype=t_sgm.dtype, device=device)),torch.tensor(1.0, dtype=t_sgm.dtype, device=device))
                # plt.hist(np.sort(t_sgm.cpu().numpy()), bins=50) 
                # plt.savefig(f"figs/{seed}_{i_cod}.png")
                # plt.clf()
                
                # with torch.no_grad():

                    # snn.eval() 
                    # t_sgm = snn(t_xgb.to(torch.float32))
                    # t_sgm = snn(t_nn)
                
                i_max         = c_act.max()
                t_msk         = torch.arange(i_max, device=device).unsqueeze(0) < c_act.unsqueeze(1)
                t_mtx         = torch.zeros_like(t_msk, dtype=t_sgm.dtype)
                t_mtx[t_msk]  = t_sgm
                t_smp         = torch.multinomial(t_mtx, num_samples=1).squeeze(1) 
                t_gps         = torch.cat([torch.tensor([0], device=device), c_act.cumsum(0)[:-1]]) # group starts 
                
                t_exs         = torch.zeros_like(t_sgm, dtype=torch.bool, device=device)
                t_exs[t_gps + t_smp] = True 

                t_inf         = t_inf[t_exs]
                t_snw         = t_snw[t_exs]
            
            
                if to_latex:

                    t_ltx['t_act_'+i_cod] =  t_act[t_exs].clone()
                    t_ltx['t_snw_'+i_cod] =  t_snw.squeeze(0).clone()

        

        t_snw[:,d_snw['pts_dlt']]   = t_snw[:,d_snw['a_pts']] + t_snw[:,d_snw['a_sur']] - t_snw[:,d_snw['b_pts']] - t_snw[:,d_snw['b_sur']]
        
        t_scr[:,:3] = t_scr[:,:3]+t_snw[:,:3].clone()
        t_mal = t_scr[:,d_scr['a_clb']]>=7
        t_mbb = t_scr[:,d_scr['b_clb']]>=7
        t_scr[:,d_scr['max_clb']][t_mal]= 1
        t_scr[:,d_scr['max_clb']][t_mbb]= 2
        t_mcl = t_scr[:,d_scr['max_clb']]>0
        t_scr[:,0:2].masked_fill_(t_mcl.unsqueeze(-1), 0)

        if i_hnd ==  5: 
            t_inf, t_snw = cleanpool(t_inf, t_snw, d_msk)
            if to_latex: 
                t_ltx['t_snw_6'] = t_snw.squeeze(0).clone()


        t_inf        = t_inf[:,0,:].unsqueeze(1)
        t_inf = F.pad(t_inf, (0, 0, 0, 2))


    t_fsc = 7*(2*(t_scr[:,-1] % 2)-1)+t_scr[:,-2]
    t_win = (t_fsc>=0).sum()/t_fsc.shape[0]
    # t_win = t_fsc.sum()/t_fsc.shape[0]
    # import pdb; pdb.set_trace()
    return t_win, t_ltx

if __name__ == "__main__":
    
    # for model in range(1):

    booster = xgb.Booster()
    booster.load_model(f"../XGB/xgb_model.json")

    # Generate random features and labels
    X = np.random.rand(100, 56)  # 100 samples, 56 features
    y = np.random.rand(100)      # 100 target values

    # Create DMatrix
    dtrain = xgb.DMatrix(X, label=y)
    params = {
    "objective": "reg:squarederror",
    "base_score": 0.0,
    "tree_method": "hist",
    "predictor": "cpu_predictor"}
    

    xrnn_alx = xgb.train(params, dtrain=dtrain, num_boost_round=0)
    xrnn_bob = xgb.train(params, dtrain=dtrain, num_boost_round=0)

    i_gin = 0
    seeds = torch.randint(0, 2**32, (10,), dtype=torch.int64)

    with tqdm(seeds, desc="SelfPlaying:") as pbar:

        for seed in pbar:

            t0_win, _ = selfplay(seed=seed, N=1000, to_latex=True, x_alx=booster,  x_bob=xrnn_bob)
            # t1_win, _ = selfplay(seed=seed, N=10000, to_latex=False, x_alx=xrnn_alx, x_bob=xrnn_bob)

            # i_gin += (t0_win - t1_win).item()
            i_gin   += t0_win.item() 
            pbar.set_postfix(gain=f"{i_gin / (pbar.n + 1):.4f}")

    # i_inp = 4*52+4
    # layer_dims = [i_inp, 64, 32,1]
    # snn_alx = SNN(layer_dims, activation=nn.Sigmoid, dropout_p=0.3).to(device).apply(init_weights_zero)
    # snn_bob = SNN(layer_dims, activation=nn.Sigmoid, dropout_p=0.3).to(device).apply(init_weights_zero)
    # t_win, t_ltx = selfplay(seed=10, N=100000, to_latex = False, snn_alx = snn_alx, snn_bob=snn_bob)
    # import pdb; pdb.set_trace()
    # print("Win Percentage:", )



# i_input  = 4*52+4