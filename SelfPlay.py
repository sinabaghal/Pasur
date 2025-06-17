import torch, sys, os, math
import torch.nn.functional as F
from FindMoves import find_moves
from ApplyMoves import apply_moves
from Imports import device, INT8, INT32, d_snw, d_scr
from CleanPool import cleanpool 
# from NeuralNet import SNN, init_weights_zero
# import torch.nn as nn
from Utils import pad_helper
from Imports import gammas, gamma_ids
# from tqdm import trange
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
# from tqdm import trange, tqdm
# import numpy as np 
# from zstd_store import load_tensor 
# import matplotlib.pyplot as plt


def playrandom(t_dck, N=1, x_alx = 'random', x_bob = 'random', to_latex = False):


    t_m52 = torch.tensor([True for i in range(52)], device=device)
    find_dm = lambda lm :  torch.tensor([idx in lm for idx in torch.arange(52, device=device)], dtype = bool, device=device)
    # find = lambda t_s, t_l: (t_s[:, None, :] == t_l[None, :, :]).all(dim=2).float().argmax(dim=1)

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

    t_ltx   = {}

    
    
    # t_dck   = t_dck.repeat(N,1)
    clt_dck = t_dck.clone()
    t_inf   = torch.zeros((N,3,52), dtype=INT8, device=device)
    # t_idx   = torch.arange(t_inf.shape[0], device=t_dck.device).unsqueeze(1).expand(-1, 4)
    # t_inf[t_idx,0,t_dck[:,:4]] = 3
    t_inf[:,0,t_dck[:4]] = 3

    t_dck  = t_dck[4:]
    # t_idx  = torch.arange(t_inf.shape[0], device=t_dck.device).unsqueeze(1).expand(-1, 4)
    t_nnd  = torch.zeros((1,52), dtype =torch.float32, device=device)
    t_scr  = torch.zeros((t_inf.shape[0],len(d_scr)), device=device,dtype=INT8)
    
    

    for i_hnd in range(6):

        t_snw     = torch.zeros((t_inf.shape[0],len(d_snw)), device=device,dtype=INT8)
        tc_scr    = t_scr[:,[0,1,3]].clone()
        
        if i_hnd > 0: 
            t_nnd[:,clt_dck[:8*i_hnd+4]] = 1
        # torch.cuda.empty_cache()
        t_inf[:,0, t_dck[:4]]  = 1
        t_inf[:,0, t_dck[4:8]] = 2
        t_dck                        = t_dck[8:]
        
        if to_latex: t_ltx[f't_scr_{i_hnd}'] = t_scr.squeeze(0).clone()

        for i_trn in range(4):
            for i_ply in range(2):
                
                i_cod = f'{i_hnd}_{i_trn}_{i_ply}'
                # print(i_cod, t_inf.shape[0])

                t_act, c_act  = find_moves(t_inf,i_ply, d_msk, d_pad) 


                if to_latex: t_ltx['t_dck_'+i_cod] = t_inf[:,0,:].clone()

                t_inf          = torch.repeat_interleave(t_inf, c_act, dim=0)
                t_snw          = torch.repeat_interleave(t_snw, c_act, dim=0)
                t_inf, t_snw   = apply_moves(t_inf, t_act, t_snw, d_msk,  i_hnd, i_ply, i_trn)


                # t_prq = torch.zeros((t_sid.shape[0], 60), dtype=torch.int8, device=device)
                # t_prq[:,t_m52]       = t_mdl[t_sid[:,0]]
                # t_prq[:,53:56]       = t_scr[t_sid[:,1]] #53,54,55
                # t_prq[:,56:59]       = torch.tensor([i_hnd,i_trn,i_ply], dtype=torch.int8, device=device) #56,57,58

                # t_prq[torch.logical_and(F.pad(t_nnd, (0, 8))==1, t_prq==0)] = -127
                #  A	B	W	H	T	P	D
                t_xgl                 = t_inf[:,1,:]-t_inf[:,2,:]
                t_xgl[torch.logical_and(t_inf[:,0,:]==3, t_xgl==0)] = 110        ### Card was already in the pool 
                t_xgl[torch.logical_and(t_inf[:,0,:]==i_ply+1, t_xgl==0)] = 100  ### Player has the card

                t_xgb          = torch.zeros((t_inf.shape[0],58), dtype=torch.int8, device=device)
                t_xgb[:,torch.cat([t_m52,torch.tensor([False,False,False,False,False,False], device=device)],dim=0)] = t_xgl
                t_xgb[torch.logical_and(F.pad(t_nnd, (0, 6))==1, t_xgb==0)] = -127
                t_xgb[:,52:55] = torch.repeat_interleave(tc_scr, c_act, dim=0)
                t_xgb[:,55:58] = torch.tensor([i_hnd,i_trn,i_ply], dtype=torch.int8, device=device)
                t_xgb_np = t_xgb.cpu().numpy()  
                n_xgb = xgb.DMatrix(t_xgb_np)  
                
                
                # t_xgb[:,52:] = t_scr[t_sid[:,1]]


                # t_xgl                 = t_inf[:,1,:]-t_inf[:,2,:]
                # t_xgl[torch.logical_and(t_inf[:,0,:]==3, t_xgl==0)] = 110  ## already in pool
                # t_xgl[torch.logical_and(t_inf[:,0,:]==i_ply+1, t_xgl==0)] = 100 ## holds

                # t_xgb          = torch.zeros((t_inf.shape[0],56), dtype=torch.int8, device=device)
                # t_xgb[:,torch.cat([t_m52,torch.tensor([False,False,False,False], device=device)],dim=0)] = t_xgl
                # t_xgb[torch.logical_and(F.pad(t_nnd, (0, 4))==1, t_xgb==0)] = -127
                

               
                i_max         = c_act.max()
                t_msk         = torch.arange(i_max, device=device).unsqueeze(0) < c_act.unsqueeze(1)
                t_mtx         = torch.zeros_like(t_msk, device=device, dtype=torch.float32)

                if i_ply == 0:
                    if x_alx == 'random':

                        t_mtx[t_msk]  = torch.tensor(1,dtype=torch.float32, device=device) 
                    else:
                        t_mtx[t_msk]  = torch.clamp_min(torch.from_numpy(x_alx.predict(n_xgb)).to(device),0)
                else:
                    if x_bob == 'random':
                        
                        t_mtx[t_msk]  = torch.tensor(1,dtype=torch.float32, device=device) 
                    else:
                        t_mtx[t_msk]  = torch.clamp_min(torch.from_numpy(x_bob.predict(n_xgb)).to(device),0)

                
                t_smp         = torch.multinomial(t_mtx, num_samples=1).squeeze(1) 

                t_gps         = torch.cat([torch.tensor([0], device=device), c_act.cumsum(0)[:-1]]) # group starts 
                m_exs         = torch.zeros(t_inf.shape[0], dtype=torch.bool, device=device)
                
                m_exs[t_gps + t_smp] = True 
                
                t_inf         = t_inf[m_exs]
                t_snw         = t_snw[m_exs]
               

               
                if to_latex:

                    t_ltx['t_act_'+i_cod] =  t_act[m_exs].clone()
                    t_ltx['t_snw_'+i_cod] =  t_snw.squeeze(0).clone()

        

        t_snw[:,d_snw['pts_dlt']]   = t_snw[:,d_snw['a_pts']] + t_snw[:,d_snw['a_sur']] - t_snw[:,d_snw['b_pts']] - t_snw[:,d_snw['b_sur']]
        
        if i_hnd ==  5: 
            t_inf, t_snw = cleanpool(t_inf, t_snw, d_msk)
            # if i_hnd ==  5: import pdb; pdb.set_trace()
            if to_latex: 
                t_ltx['t_snw_6'] = t_snw.squeeze(0).clone()

        t_scr[:,:3] = t_scr[:,:3]+t_snw[:,:3].clone()
        # if i_hnd ==  5: import pdb; pdb.set_trace()
        t_mal = t_scr[:,d_scr['a_clb']]>=7
        t_mbb = t_scr[:,d_scr['b_clb']]>=7
        t_scr[:,d_scr['max_clb']][t_mal]= 1
        t_scr[:,d_scr['max_clb']][t_mbb]= 2
        t_mcl = t_scr[:,d_scr['max_clb']]>0
        t_scr[:,0:2].masked_fill_(t_mcl.unsqueeze(-1), 0)


        t_inf = t_inf[:,0,:].unsqueeze(1)
        t_inf = F.pad(t_inf, (0, 0, 0, 2))


    t_fsc = 7*(2*(t_scr[:,-1] % 2)-1)+t_scr[:,-2]
    # t_win = (t_fsc>=0).sum()/t_fsc.shape[0]
    return t_fsc, t_ltx



if __name__ == "__main__":

    N = 10000
    csv_file = '../MDL/scores_log.csv'
    t_dks = torch.load(f"decks_10000.pt")

    if not os.path.isfile(csv_file):
        pd.DataFrame(columns=['DECK', 'Alex_Gamma', 'Bob_Gamma', 'Win Rate']).to_csv(csv_file, index=False)

    def load_model(id_dck, id_gamma):

        bst = xgb.Booster()
        bst.load_model(f'../MDL/D{id_dck}/model_{id_dck}_{id_gamma}.xgb')
        return bst 
    
    # gamma_ids = [0,1,10]
    for i_dck in [9]:

        models = {id_gamma:load_model(i_dck, id_gamma) for id_gamma in gamma_ids}
        models[-1] = 'random'
        
        t_dck = t_dks[i_dck,:].to(device)

        scores = {}
        # gamma_ids + [-1]
        for id_gamma_1 in [0,-1]:
            for id_gamma_2 in [0,-1]:

                t_fsc, _ = playrandom(t_dck, N=N, x_alx = models[id_gamma_1] , x_bob = models[id_gamma_2])
                
                scores[(id_gamma_1, id_gamma_2)] = (t_fsc>0).sum().item() / t_fsc.shape[0]
                

        df = pd.DataFrame([{'DECK': i_dck, 'Alex_Gamma': k[0], 'Bob_Gamma': k[1], 'Win Rate': v} for k, v in scores.items()])
        df.to_csv(csv_file, mode='a', header=False, index=False)


        # csv_file = '../MDL/scores_log.csv'
        # fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(24, 4))  # 7 heatmaps in a row

    
        
    
    # bst10 = xgb.Booster()
    # bst10.load_model('../MDL/D8/model_8_10.xgb')
    
    # bst100 = xgb.Booster()
    # bst100.load_model('../MDL/D8/model_8_100.xgb')



    
    
    # Ks = list(range(int(N/100), N+1, 100))
    # dict = {}

    # plt.clf()

    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = bst0 , x_bob = 'random')
    # dict['0vsr'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]
    
    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = 'random' , x_bob = bst0)
    # dict['rvs0'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]

    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = bst10 , x_bob = 'random')
    # dict['10vsr'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]

    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = 'random' , x_bob = bst10)
    # dict['rvs10'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]

    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = bst100 , x_bob = 'random')
    # dict['100vsr'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]
    
    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = 'random' , x_bob = bst100)
    # dict['rvs100'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]
    
    # ######## 100 vs 0 ###############    
    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = bst0 , x_bob = bst100)
    # dict['0vs100'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]

    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = bst100 , x_bob = bst0)
    # dict['100vs0'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]

    # ######## 100 vs 10 ###############    
    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = bst10 , x_bob = bst100)
    # dict['10vs100'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]

    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = bst100 , x_bob = bst10)
    # dict['100vs10'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]

    # ######## 0 vs 10 ###############    
    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = bst0 , x_bob = bst10)
    # dict['0vs10'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]
    
    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = bst10, x_bob = bst0)
    # dict['10vs0'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]
    



    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = bst10 , x_bob = bst10)
    # dict['10vs10'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]
 
    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = bst100 , x_bob = bst100)
    # dict['100vs100'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]
    
    # t_fsc, _ = playrandom(t_dck, N=N, x_alx = bst0 , x_bob = bst0)
    # dict['0vs0'] = [(t_fsc[:K]>0).sum().item() / K for K in Ks]

    # def plot(keys):

    #     for key in keys:
            
    #         plt.plot(Ks, dict[key], label=key)

    #     plt.legend()
    #     plt.show()
    

    # def plotpt(values, labels):

    #     plt.figure(figsize=(6, 2))
    #     y = [1] * len(values)
    #     plt.plot(values, y, 'o')

    #     # Add vertical labels below each point
    #     for x, label in zip(values, labels):
    #         plt.text(x, 0.99, label, fontsize=8, ha='center', va='top', rotation=90)

    #     plt.yticks([])  # Hide y-axis
    #     plt.xlim(0, 1.05)
    #     plt.ylim(0.9, 1.05)
    #     plt.title("Labeled Values (Vertical Below)")
    #     plt.grid(True, axis='x')
    #     plt.show()




    # keys = dict.keys()
    # values = [dict[key][-1] for key in keys]







    

  
    
