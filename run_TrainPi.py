from Imports   import device
from NeuralNet import SNN, init_weights_zero
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn
import torch, os, shutil
import gc, random  
from zstd_store import load_tensor 
import torch.optim as optim
from SelfPlay  import selfplay
from tqdm import trange, tqdm





# @profile 
def run():

    files = os.listdir(f"../STRG/sgm/")
    best_loss = float('inf')

    size = 24
    bi_sze = 512
    lr=1e-3
    i_itr = 100

    nn_dims = [56, 32,8,1]
    pbar = trange(i_itr)

    snn = SNN(nn_dims, activation=nn.Sigmoid, dropout_p=0.95).to(device)
    optimizer = optim.Adam(snn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.MSELoss()
    # snn_dtype = snn.layers[0].weight.dtype


    for epoch in pbar:

        random.shuffle(files)
        files_list = [files[i:i+size] for i in range(0, len(files), size)]

        for these_files in files_list:


            b_fct = torch.sqrt(torch.tensor([int(folder.split('_')[0][3:])+1 for folder in these_files], device=device))

            with ThreadPoolExecutor() as executor:

                futures_sgm = [executor.submit(load_tensor, f"../STRG/sgm/{filepath}") for filepath in these_files]
                bt_sgm = [f.result() for f in futures_sgm]

            with ThreadPoolExecutor() as executor:

                futures_nns = [executor.submit(load_tensor, f"../STRG/nns/{filepath}") for filepath in these_files]
                bt_nns = [f.result() for f in futures_nns]

            b_fct   =  torch.repeat_interleave(b_fct,torch.tensor([tensor.shape[0] for tensor in bt_nns], device=device))
            t_sgm   = torch.cat(bt_sgm, dim=0).to(device)
            t_nns   = torch.cat(bt_nns, dim=0).to(device, dtype=torch.float32) 
            
            del bt_sgm, bt_nns 
            
            tot_loss = 0.0 

            for i_beg in range(0, t_nns.shape[0], bi_sze):

                snn.train()

                i_end = i_beg + bi_sze
                b_nns = b_fct[i_beg:i_end]*snn(t_nns[i_beg:i_end])
                b_sgm = b_fct[i_beg:i_end]*t_sgm[i_beg:i_end]
                
                t_lss = criterion(b_nns, b_sgm)
                
                optimizer.zero_grad()
                t_lss.backward()
                optimizer.step()
                scheduler.step()
                
                snn.eval()
                with torch.no_grad():

                    loss = criterion(b_nns, b_sgm)*bi_sze
                    tot_loss += loss 
            
            
            
            tot_loss /= t_nns.shape[0]


        if tot_loss < best_loss:
            best_loss = tot_loss.item()
        
        pbar.set_postfix(loss=tot_loss.item(), best_loss=best_loss)

run()
# import pdb; pdb.set_trace()
# srnn_alx = SNN(nn_dims, activation=nn.Sigmoid, dropout_p=0.0).to(device).apply(init_weights_zero)
# srnn_bob = SNN(nn_dims, activation=nn.Sigmoid, dropout_p=0.0).to(device).apply(init_weights_zero)

# i_gin = 0
# seeds = torch.randint(0, 2**32, (20,), dtype=torch.int64)

# with tqdm(seeds, desc="SelfPlaying:") as pbar:

#     for seed in pbar:

#         t0_win, _ = selfplay(seed=seed, N=10000, to_latex=False, snn_alx=snn, snn_bob=srnn_bob)
#         t1_win, _ = selfplay(seed=seed, N=10000, to_latex=False, snn_alx=srnn_alx, snn_bob=srnn_bob)

#         i_gin += (t0_win - t1_win).item()
#         pbar.set_postfix(gain=f"{i_gin / (pbar.n + 1):.4f}")