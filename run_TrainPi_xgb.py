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
import numpy as np 
import xgboost as xgb
from Utils import partition_folder

import numpy as np

# def custom_obj(y_pred, dtrain):

#     y_true = dtrain.get_label()

#     grad = y_pred - y_true
#     hess = np.ones_like(y_pred)

#     penalty_factor = 10  # You can tune this value

#     # Penalize predictions < 0
#     mask_low = y_pred < 0
#     grad[mask_low] += penalty_factor * y_pred[mask_low]
#     hess[mask_low] += penalty_factor

#     # Penalize predictions > 1
#     mask_high = y_pred > 1
#     grad[mask_high] += penalty_factor * (y_pred[mask_high] - 1)
#     hess[mask_high] += penalty_factor

#     return grad, hess



# @profile 
def run(save_folder=None, booster = None, MAX_FILES=50, MAX_TOTAL_SIZE_MB=50):

    # files = os.listdir(f"../STRG/sgm/")
    # size = 2

    params = {
        "max_depth": 12, 
        "objective": "reg:squarederror",
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "verbosity": 0,
        "eta":0.3
    }

    
    num_rounds = 500

    # snn_dtype = snn.layers[0].weight.dtype

    # files_list = partition_folder(,limit)
    # files_list = partition_folder(f"{save_folder}/STRG/sgm/", MAX_FILES=MAX_FILES, MAX_TOTAL_SIZE_MB=MAX_TOTAL_SIZE_MB)
    files_list = partition_folder(f"{save_folder}/STRG/nns/", MAX_FILES=MAX_FILES, MAX_TOTAL_SIZE_MB=MAX_TOTAL_SIZE_MB)
    files_list = [[file.split('/')[-1] for file in files] for files in files_list]
    
    # random.shuffle(files)
    # files_list = [files[i:i+size] for i in range(0, len(files), size)]

    for these_files in tqdm(files_list):

        torch.cuda.empty_cache()
        b_fct = torch.sqrt(torch.tensor([int(folder.split('_')[0][3:])+1 for folder in these_files], device=device))

        with ThreadPoolExecutor() as executor:

            futures_sgm = [executor.submit(load_tensor, f"{save_folder}/STRG/sgm/{filepath}") for filepath in these_files]
            bt_sgm = [f.result() for f in futures_sgm]

        with ThreadPoolExecutor() as executor:

            futures_nns = [executor.submit(load_tensor, f"{save_folder}/STRG/nns/{filepath}") for filepath in these_files]
            bt_nns = [f.result() for f in futures_nns]

        
        # def create_batches(X, y, weight, batch_size):
        #     n = X.shape[0]
        #     for i in range(0, n, batch_size):
        #         yield X[i:i+batch_size], y[i:i+batch_size], weight[i:i+batch_size]

        # Prepare data
        X      = torch.cat(bt_nns).numpy()               # shape: (N, 56)
        y      = torch.cat(bt_sgm).cpu().numpy()         # shape: (N,)

        b_fct  = np.sqrt(torch.tensor([int(folder.split('_')[0][3:]) + 1 for folder in these_files]))
        b_fct  = b_fct/b_fct.max()
        # import pdb; pdb.set_trace()
        
        weight = np.repeat(b_fct, np.array([x.shape[0] for x in bt_nns]))  # shape: (N,)
        del bt_nns, bt_sgm, b_fct

        dtrain = xgb.DMatrix(X, label=y, weight=weight)
        # import pdb; pdb.set_trace()
        # del X, y, weight
        gc.collect()
        evals = [(dtrain, "train")]

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_rounds,
            xgb_model=booster,
            # obj=custom_obj,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # y_pred = booster.predict(dtrain)
        # import pdb; pdb.set_trace()


        # Batch and train
        # data_batches = create_batches(X, y, weight, batch_size= 800000)

        # for X_batch, y_batch, weight_batch in data_batches:
        #     dtrain = xgb.DMatrix(X_batch, label=y_batch, weight=weight_batch)
        #     del X_batch, y_batch, weight_batch
        #     gc.collect()

        #     evals = [(dtrain, "train")]

        #     booster = xgb.train(
        #         params,
        #         dtrain,
        #         num_boost_round=num_rounds,
        #         xgb_model=booster,
        #         obj=custom_obj,
        #         evals=evals,
        #         early_stopping_rounds=10,
        #         verbose_eval=False
        #     )

    
    return booster
            


# booster = xgb.Booster()
save_folder = 'D:/Pasur'    
booster = None
booster = run(save_folder=save_folder, booster=booster, MAX_FILES=100, MAX_TOTAL_SIZE_MB=50)
booster.save_model(f"../XGB/xgb_model.json")
# import pdb; pdb.set_trace()

# # booster = xgb.Booster()
# # booster.load_model("xgb_model.json")

# # xrnn_alx = SNN(nn_dims, activation=nn.Sigmoid, dropout_p=0.0).to(device).apply(init_weights_zero)
# # xrnn_alx = SNN(nn_dims, activation=nn.Sigmoid, dropout_p=0.0).to(device).apply(init_weights_zero)

# dtest = xgb.DMatrix(np.random.rand(3, 56).astype(np.float32))
# params = {
# "objective": "reg:squarederror",
# "base_score": 0.0,
# "tree_method": "hist",
# "predictor": "cpu_predictor"}
    
# xrnn_alx = xgb.train(params, dtrain=dtest, num_boost_round=0)
# xrnn_bob = xgb.train(params, dtrain=dtest, num_boost_round=0)

# i_gin = 0
# seeds = torch.randint(0, 2**32, (20,), dtype=torch.int64)

# with tqdm(seeds, desc="SelfPlaying:") as pbar:

#     for seed in pbar:

#         t0_win, _ = selfplay(seed=seed, N=10000, to_latex=False, x_alx=booster,  x_bob=xrnn_bob)
#         t1_win, _ = selfplay(seed=seed, N=10000, to_latex=False, x_alx=xrnn_alx, x_bob=xrnn_bob)

#         i_gin += (t0_win - t1_win).item()
#         pbar.set_postfix(gain=f"{i_gin / (pbar.n + 1):.4f}")