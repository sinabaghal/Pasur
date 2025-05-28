import torch, os, copy,  shutil, time
import xgboost as xgb
from Imports import device
from NeuralNet import NN,init_weights_zero
from SimulateGameRun import simulategamerun
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from tqdm import trange
from zstd_store import to_memory, load_tensor 
import torch
import numpy as np 
from concurrent.futures import ThreadPoolExecutor
from Utils import partition
import random 
import gc
from Utils import partition_folder


# @profile 
def deepcfr(OFFSET=100, CFR=3, TRV=10, save_folder = None, NUM_ROUNDS = 50, MAX_FILES=100, MAX_TOTAL_SIZE_MB=50):


    find_folder = lambda i_cfr, i_trv, i_ext : f'CFR{OFFSET+i_cfr}_TRV{i_trv}_EXT{i_ext}'

    dtest = xgb.DMatrix(np.random.rand(3, 56).astype(np.float32))


    params = {
    "max_depth": 12, 
    "objective": "reg:squarederror",
    "base_score": 0.0,
    "tree_method": "hist",
    "predictor": "cpu_predictor",
    "eta":0.5}
    
    x_alx = xgb.train(params, dtrain=dtest, num_boost_round=0)
    x_bob = xgb.train(params, dtrain=dtest, num_boost_round=0)
    

    for i_cfr in trange(CFR):
        torch.cuda.empty_cache()

        if i_cfr != 0 and i_cfr % 100 == 0:
            time.sleep(300)  

        for i_ext in range(2):

            i_oth = (i_ext+1) % 2

            for i_trv in range(TRV):
                
                folder = find_folder(i_cfr, i_trv, i_ext)
                # CFR =  int(folder.split('_')[0][3:])
                seed = int.from_bytes(os.urandom(4), byteorder="little")
                
                with torch.no_grad():

                    # str_memory, adv_memory, d_fls = simulategamerun(seed, folder, nn_alx, nn_bob, i_ext, str_memory=str_memory,adv_memory=adv_memory)
                    d_fls = simulategamerun(seed, folder, x_alx, x_bob, i_ext,  save_folder = 'D:Pasur')
                
                # simulategamerun(saving, seed, folder, nn_alx, nn_bob, i_ext)

                t_scr, t_sid = d_fls['t_scr_6'], d_fls['t_sid_6']
                t_fsc = 7*(2*(t_scr[:,-1] % 2)-1)+t_scr[:,-2]
                t_val = t_fsc[t_sid[:,1]].to(torch.float32)
                t_regs = {}
                for i_hnd in reversed(range(6)):

                    t_lnk  = d_fls[f't_lnk_{i_hnd}'] 
                    t_val  = t_val[t_lnk]
                    
                    for i_trn in reversed(range(4)):
                        for i_ply in reversed(range(2)):

                            i_cod        = f'{i_hnd}_{i_trn}_{i_ply}'
                            
                            if i_ply == i_ext:

                                t_edg = d_fls[f't_edg_{i_cod}']
                                t_val = t_val[t_edg]

                            else:

                                i_sid        = d_fls[f'i_sid_{i_cod}'] 
                                t_sum        = torch.zeros(i_sid, dtype = t_val.dtype, device=device)  
                                t_edg, t_sgm = d_fls[f't_edg_{i_cod}'], d_fls[f't_sgm_{i_cod}']
                                t_sum.scatter_add_(0, t_edg, t_sgm*t_val)
                                t_reg  = t_val - t_sum[t_edg]
                                t_regs[i_cod] = t_reg
                                t_val  = t_sum

                adv_keys = [
                f'{i_hnd}_{i_trn}_{i_oth}'
                for i_hnd in range(6)
                for i_trn in range(4)
                ]

                t_regs = torch.cat([t_regs[id] for id in adv_keys] ,0).cpu()
                # adv_memory[folder+'_reg'] = t_regs.to('cpu').numpy()
                to_memory(t_regs,  f"{save_folder}/ADVT/reg/{folder}.pt.zst")

                # to_memory(t_regs, folder+'_Adv_t_reg.zstd', device='cpu', memory_store = adv_memory)



            params = {
                "max_depth": 12, 
                "objective": "reg:squarederror",
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "verbosity": 0
            }

            booster = None
            num_rounds = NUM_ROUNDS

            files_list = partition_folder(f"{save_folder}/ADVT/nns/", MAX_FILES=MAX_FILES, MAX_TOTAL_SIZE_MB=MAX_TOTAL_SIZE_MB)
            files_list = [[file.split('/')[-1] for file in files] for files in files_list]

            for these_files in files_list:

                torch.cuda.empty_cache()

                with ThreadPoolExecutor() as executor:

                    futures_nns = [executor.submit(load_tensor, f"{save_folder}/ADVT/nns/{filepath}") for filepath in these_files]
                    bt_nns = [f.result() for f in futures_nns]
                    # bt_nns = list(executor.map(load_numpy, [f"../ADVT/nns/{filepath}" for filepath in these_files]))

                with ThreadPoolExecutor() as executor:

                    futures_reg = [executor.submit(load_tensor, f"{save_folder}/ADVT/reg/{filepath}") for filepath in these_files]
                    bt_reg = [f.result() for f in futures_reg]
                
                # Xs = [adv_memory[folder+'_nn'] for folder in b_hst]
                X      = torch.cat(bt_nns).numpy()
                y      = torch.cat(bt_reg).numpy()
                y = y/40

                b_fct  = np.sqrt(torch.tensor([int(folder.split('_')[0][3:])+1 for folder in these_files]))
                b_fct  = b_fct/b_fct.max()
                weight = np.repeat(b_fct,np.array([x.shape[0] for x in bt_nns]))

                # del 

                # import pdb; pdb.set_trace()
                dtrain = xgb.DMatrix(X, label=y,  weight=weight)
                # print('MIN =', y.min(), 'MAX=', y.max())
                del X, y, weight, bt_nns, bt_reg
                gc.collect()  # Optional: run garbage collector
                
                evals = [(dtrain, "train")]

                booster = xgb.train(
                params,
                dtrain,
                num_boost_round=num_rounds,
                xgb_model=booster, 
                evals=evals, 
                early_stopping_rounds=50,
                verbose_eval=False)

            
            if i_oth == 0: x_alx  = booster
            else:          x_bob  = booster


if __name__ == "__main__":


    CFR = 1
    TRV = 1
    OFFSET = 0
    save_folder = 'D:Pasur'
    
    os.makedirs(f"{save_folder}/STRG/nns/", exist_ok=True)
    os.makedirs(f"{save_folder}/STRG/sgm/", exist_ok=True)
    
    os.makedirs(f"{save_folder}/ADVT/nns/", exist_ok=True)
    os.makedirs(f"{save_folder}/ADVT/reg/", exist_ok=True)

    ### CONSTRUCT STRATEGY MEMORY
    print("\n"+50*"*" + "  DEEP CFR  " + 50*"*"+"\n")
    deepcfr(OFFSET=OFFSET, CFR=CFR, TRV=TRV, save_folder=save_folder, NUM_ROUNDS=100, MAX_FILES=100, MAX_TOTAL_SIZE_MB=70)