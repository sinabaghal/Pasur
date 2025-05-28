import torch 
from NeuralNet import SNN
from Imports import device
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from zstd_store import load_memory
from concurrent.futures import ThreadPoolExecutor
import copy, random
import numpy as np
# @profile
def trainpi(CFR=3, TRV=2, bi_sze = 512, i_itr = 10, i_c2g = 2, str_memory=None, nn_dims = [4*52+4, 64, 32,1], lr=1e-3):

    find_folder = lambda i_cfr, i_trv, i_ext : f'CFR{i_cfr}_TRV{i_trv}_EXT{i_ext}'

    snn = SNN(nn_dims, activation=nn.Sigmoid).to(device)
    optimizer = optim.Adam(snn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.MSELoss()

    snn_dtype = snn.layers[0].weight.dtype
    snn.train()

    pbar = trange(i_itr)
    best_loss = float('inf')
    best_model_state = None
        
    # t_fct = torch.sqrt(torch.tensor([i_cfr + 1 
    #                                 for i_ext in range(2) 
    #                                 for i_trv in range(TRV) 
    #                                 for i_cfr in range(CFR)], 
    #                                 dtype=snn_dtype, device=device))


    ### SIGMAS

    hist = [find_folder(i_cfr, i_trv, i_ext)
            for i_ext in range(2)
            for i_trv in range(TRV)
            for i_cfr in range(CFR)]

    # tasks_sgm = [
    #     (find_folder(i_cfr, i_trv, i_ext)+'_Str_t_sgm.zstd', str_memory)
    #     for i_ext in range(2)
    #     for i_trv in range(TRV)
    #     for i_cfr in range(CFR)
    # ]

    # with ThreadPoolExecutor() as executor:
 
    #     futures = [executor.submit(load_memory, *args) for args in tasks_sgm]
    #     t_sgm   = [f.result() for f in futures]

    # t_fct  = torch.repeat_interleave(t_fct, torch.tensor([tensor.shape[0] for tensor in t_sgm], device=device))
    # t_sgm  = t_fct*(torch.cat(t_sgm, dim=0).to(device).to(snn_dtype))

    # ### INFOSETS 

    # tasks_nns = [
    #     (find_folder(i_cfr, i_trv, i_ext)+'_Str_t_nn.zstd', str_memory)
    #     for i_ext in range(2)
    #     for i_trv in range(TRV)
    #     for i_cfr in range(CFR)
    # ]

    # with ThreadPoolExecutor() as executor:

    #     futures = [executor.submit(load_memory, *args) for args in tasks_nns]
    #     t_cnn   = [f.result() for f in futures]


    # t_cnn = torch.cat(t_cnn, dim=0)
    for epoch in pbar:

        random.shuffle(hist)
        hists = [hist[i:i+i_c2g] for i in range(0, len(hist), i_c2g)]

        tot_loss = 0.0 
        i_num = 0
        

        for b_hst in hists:


            # X = np.concatenate([str_memory[folder+'_nn'] for folder in b_hst])
            # y = np.concatenate([str_memory[folder+'_reg'] for folder in b_hst])


            # b_nns = [ (folder+'_nn',  str_memory) for folder in b_hst]
            # b_sgm = [ (folder+'_sgm', str_memory) for folder in b_hst]
            b_fct       = torch.sqrt(torch.tensor([int(folder.split('_')[0][3:])+1 for folder in b_hst], device=device))

            bt_nns      = [torch.from_numpy(str_memory[folder+'_nn']) for folder in b_hst]
            bt_sgm      = [torch.from_numpy(str_memory[folder+'_sgm']) for folder in b_hst]
            
            
            b_fct       = torch.repeat_interleave(b_fct, torch.tensor([tensor.shape[0] for tensor in bt_nns], device=device))
            bt_nns      =  b_fct*snn(torch.cat(bt_nns, dim=0).to(device=device, dtype=snn_dtype))
            bt_sgm      =  b_fct*torch.cat(bt_sgm, dim=0).to(device=device, dtype=snn_dtype)
            # bt_sgm      = b_fct*bt_sgm
            
            # with ThreadPoolExecutor() as executor:
                
            #     futures  = [executor.submit(load_memory, *args) for args in b_nns]
            #     bt_nns   = [torch.from_numpy(f.result()) for f in futures]

            # bt_nns    = torch.cat(bt_nns, dim=0).pin_memory()
            # bt_nns    = bt_nns.to(device=device, dtype=snn_dtype, non_blocking=True)
            # to(device=device, dtype=snn_dtype)
            # bt_nns    = b_fct*snn(bt_nns)
            
            # with ThreadPoolExecutor() as executor:
                
            #     futures = [executor.submit(load_memory, *args) for args in b_sgm]
            #     bt_sgm  = [f.result() for f in futures]

            # bt_sgm    = b_fct*(torch.cat(torch.from_numpy(bt_sgm), dim=0).to(device=device, dtype=snn_dtype))
            i_num    += bt_nns.shape[0]

            for i_beg in range(0, bt_nns.shape[0], bi_sze):

                i_end = i_beg + bi_sze
                b_nns = bt_nns[i_beg:i_end]
                b_sgm = b_fct[i_beg:i_end]
                t_lss = criterion(b_nns, b_sgm)*bi_sze
                tot_loss += t_lss 
                # b_nns = b_fct[i_beg:i_end]*snn(bt_nns[i_beg:i_end].to(device).to(snn_dtype))

        tot_loss /= i_num
        # perm   = torch.randperm(i_num)

        # t_fct  = t_fct[perm]
        # t_cnn  = t_cnn[perm]
        # t_sgm  = t_sgm[perm]

        


        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
        scheduler.step()

        if tot_loss < best_loss:
            
            best_loss = tot_loss.item()
            best_model_state = copy.deepcopy(snn.state_dict())     
            torch.save(snn, f"../STRG/snn_{CFR}_{epoch}.pth")   
        
        # if epoch % 10 == 0:
        pbar.set_postfix(loss=tot_loss.item(), best_loss=best_loss)

    

    return snn

