from TrainPi   import trainpi
from DeepCFR   import deepcfr
from SelfPlay  import selfplay
from Imports   import device
from NeuralNet import SNN, init_weights_zero
import torch.nn as nn
import torch, os, shutil
from tqdm import tqdm
import gc 


adv_memory = {}
str_memory = {}

CFR = 2
TRV = 2
bi_sze = 1024
nn_dims = [56, 16, 1]

if os.path.exists(f"../STRG/"): shutil.rmtree(f"../STRG/")
os.makedirs(f"../STRG/nns/", exist_ok=True)
os.makedirs(f"../STRG/sgm/", exist_ok=True)

### CONSTRUCT STRATEGY MEMORY
print("\n"+50*"*" + "  DEEP CFR  " + 50*"*"+"\n")
str_memory = deepcfr(CFR=CFR, TRV=TRV, bi_sze=bi_sze, i_itr = 100, i_c2g = 2, adv_memory = adv_memory, str_memory=str_memory, nn_dims = nn_dims, lr=1e-2)
del adv_memory
import pdb; pdb.set_trace()
gc.collect()                      
torch.cuda.empty_cache()          
torch.cuda.ipc_collect()          

### TRAIN STRATEGY NETWORK 
print("\n"+50*"*"+ "  TRAIN PI  " + 50*"*"+"\n")
snn = trainpi(CFR=CFR, TRV=TRV, bi_sze = bi_sze, i_itr = 50, i_c2g = 2, str_memory=str_memory, nn_dims = nn_dims, lr=1e-2)
del str_memory



srnn_alx = SNN(nn_dims, activation=nn.Sigmoid, dropout_p=0.0).to(device).apply(init_weights_zero)
srnn_bob = SNN(nn_dims, activation=nn.Sigmoid, dropout_p=0.0).to(device).apply(init_weights_zero)

i_gin = 0
seeds = torch.randint(0, 2**32, (2,), dtype=torch.int64)

with tqdm(seeds, desc="SelfPlaying:") as pbar:

    for seed in pbar:

        t0_win, _ = selfplay(seed=seed, N=10000, to_latex=False, snn_alx=snn, snn_bob=srnn_bob)
        t1_win, _ = selfplay(seed=seed, N=10000, to_latex=False, snn_alx=srnn_alx, snn_bob=srnn_bob)

        i_gin += (t0_win - t1_win).item()
        pbar.set_postfix(gain=f"{i_gin / (pbar.n + 1):.4f}")
