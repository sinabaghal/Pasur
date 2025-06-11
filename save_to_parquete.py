import os 
import torch 
import glob
import torch.nn.functional as F
from zstd_store import load_tensor
import pandas as pd 
from Imports import device
import pyarrow.parquet as pq
import dask.dataframe as dd
from tqdm import tqdm
import shutil
# from dask import delayed, compute
# from dask.distributed import Client
import numpy as np 

SGM_FDLR = "../DATA/SGM/"
SID_FDLR = "../DATA/SID/"
SCR_FDLR = "../DATA/SCR/"
T52_FDLR = "../DATA/T52/"
MDL_FDLR = "../DATA/MDL/"

f_sgm   = os.listdir(SGM_FDLR)
f_sid   = os.listdir(SID_FDLR)
f_scr   = os.listdir(SCR_FDLR)
f_t52   = os.listdir(T52_FDLR)
f_mdl   = os.listdir(MDL_FDLR)

i_dks = [int(file.split('_')[0][1:]) for file in f_sgm]

def get_col(i):

    if i<52:
        return f"C{i}"
    elif i == 52: return "SGM"
    elif i == 53: return "A"
    elif i == 54: return "B"
    elif i == 55: return "W"
    elif i == 56: return "H"
    elif i == 57: return "T"
    elif i == 58: return "P"
    elif i == 59: return "D"

# ddf_dict = {}
group_cols = ["A", "B", "W", "H", "T", "P"]
t_dks      = torch.load(f"decks_10000.pt")

def process_batch(i_hnd, i_trn, i_ply, i_l, i_u):

    # for i_hnd in range(6):
    #     for i_trn in range(4):
    #         for i_ply in range(2):

    #             # 'D0_scr_0.pt.zst'

    ddf_dict = {}
    for i_dck in torch.arange(start = i_l, end= i_u, dtype=torch.int32):
        if i_dck not in i_dks: continue

        t_dck = t_dks[i_dck]
        t_nnd = torch.zeros((1,52), device=device)
        
       
        if i_hnd > 0:
            
            i_nnd = 8*i_hnd+4
            t_nnd[0,t_dck[:i_nnd]] = 1

        t_sgm = load_tensor(SGM_FDLR+f'D{i_dck}_sgm_{i_hnd}_{i_trn}_{i_ply}.pt.zst').to(device=device, dtype=torch.float16)
        t_mdl = load_tensor(MDL_FDLR+f'D{i_dck}_mdl_{i_hnd}_{i_trn}_{i_ply}.pt.zst').to(device=device)
        t_sid = load_tensor(SID_FDLR+f'D{i_dck}_sid_{i_hnd}_{i_trn}_{i_ply}.pt.zst').to(device=device)
        t_m52 = load_tensor(T52_FDLR+f'D{i_dck}_t52_{i_hnd}.pt.zst').to(device=device)
        t_scr = load_tensor(SCR_FDLR+f'D{i_dck}_scr_{i_hnd}.pt.zst').to(device=device)
        
        t_m52 = torch.cat([t_m52,torch.tensor([False, False,False, False,False,False,False,False], device=device)])
        t_scr = t_scr[:,[0,1,3]]

        t_prq = torch.zeros((t_sid.shape[0], 60), dtype=torch.int8, device=device)
        t_prq[:,t_m52]       = t_mdl[t_sid[:,0]]
        t_prq[:,52]          = t_sgm
        t_prq[:,53:56]       = t_scr[t_sid[:,1]] #53,54,55
        t_prq[:,56:59]       = torch.tensor([i_hnd,i_trn,i_ply], dtype=torch.int8, device=device) #56,57,58

        t_prq[torch.logical_and(F.pad(t_nnd, (0, 8))==1, t_prq==0)] = -127
        t_prq                = t_prq.cpu()

        data = {get_col(i): t_prq[:, i] for i in range(t_prq.shape[1])}
        df   = pd.DataFrame(data)
        df["D"] = df["D"].astype("int32")
        df["D"] = np.full(len(df), i_dck.item(), dtype=np.int32)
        ddf = dd.from_pandas(df, npartitions=1)
        ddf_dict[i_dck.item()] = ddf

    ddf_all = dd.concat(list(ddf_dict.values()), npartitions=1).repartition(partition_size="100MB")
    ddf_all.to_parquet(
        f"../PRQ/{i_l}_{i_u}/{i_hnd}_{i_trn}_{i_ply}",
        engine="pyarrow",
        write_index=False,
        write_metadata_file=False  # optional, suppress _metadata if you want
    )

                    # ddf.to_parquet(
                    # f"data/iteration_{i_dck}_{i_hnd}_{i_trn}_{i_ply}/",
                    # engine="pyarrow",
                    # # partition_on=group_cols,
                    # write_index=False)
                    



                    # fs_mdl = [file for file in f_mdl if file.split('mdl_')[1].split('.')[0] == f'{i_hnd}_{i_trn}_{i_ply}' and i_l<= int(file.split('_')[0][1:]) < i_u]
                    # fs_sgm = [file for file in f_sgm if file.split('sgm_')[1].split('.')[0] == f'{i_hnd}_{i_trn}_{i_ply}' and i_l<= int(file.split('_')[0][1:]) < i_u]
                    # fs_sid = [file for file in f_sid if file.split('sid_')[1].split('.')[0] == f'{i_hnd}_{i_trn}_{i_ply}' and i_l<= int(file.split('_')[0][1:]) < i_u]

    # ddf_all = dd.concat(list(ddf_dict.values()), interleave_partitions=True).repartition(npartitions=1)
    # ddf_all.to_parquet(
    #     "merged_data/",
    #     engine="pyarrow",
    #     write_index=False,
    #     write_metadata_file=False  # optional, suppress _metadata if you want
    # )

# tasks = {}
i_cods = [
        f'{i_hnd}_{i_trn}_{i_ply}'
        for i_hnd in range(6)
        for i_trn in range(4)
        for i_ply in range(2)
        ]


# for i_cod in tqdm(i_cods):

#     i_hnd = int(i_cod.split('_')[0])
#     if i_hnd<4 : continue
    
#     folders = glob.glob(f"../PRQ/**/{i_cod}/*.parquet", recursive=True)
#     ddf_all = dd.read_parquet(folders, engine="pyarrow").repartition(npartitions=1)
#     ddf_all.to_parquet(
#     f"../MPRQ/{i_cod}",
#     engine="pyarrow",
#     write_index=False)
#     ddf_all.head(1)  # triggers computation, but is a bit hacky


for i_cod in tqdm(i_cods):

    # if i_cod != '0_0_0': continue 
     
    i_hnd, i_trn, i_ply  = [int(x) for x in i_cod.split('_')]
    process_batch(i_hnd, i_trn, i_ply, 0,200)
    # process_batch(i_hnd, i_trn, i_ply, 100,150)
    # process_batch(i_hnd, i_trn, i_ply, 150,200)

                # tasks[i_cod] = delayed(process_batch)(i_hnd, i_trn, i_ply, 0,10)
# compute(*tasks.values())
# import pdb; pdb.set_trace()
# folders = glob.glob("data/**/*.parquet", recursive=True)
# ddf_all = dd.read_parquet(folders, engine="pyarrow").repartition(npartitions=1)
# ddf_all = ddf_all.repartition(npartitions=1)
# ddf_all.to_parquet(
#     "merged_data/",
#     engine="pyarrow",
#     write_index=False
# )
# folders = glob.glob("merged_data/*.parquet", recursive=True)
# ddf = dd.read_parquet(folders, engine="pyarrow").repartition(partition_size="1GB")
# for i, part in enumerate(ddf.to_delayed()):
#     df_partition = part.compute()  # This is a Pandas DataFrame
#     import pdb; pdb.set_trace()
#     print(f"Partition {i}, shape = {df_partition.shape}")
#     # Do something with df_partition, like saving, training, etc.


# # base_path = "partitioned_data"
# # merged_base = "merged_partitioned_output"

# # # List all partition folders from all iterations
# # all_partitions = set()

# # for iter_folder in os.listdir(base_path):
# #     iter_path = os.path.join(base_path, iter_folder)
# #     # find all partition folders inside this iteration folder
# #     for root, dirs, files in os.walk(iter_path):
# #         if any(f.endswith(".parquet") for f in files):
# #             # get relative partition path after iteration folder
# #             rel_partition = os.path.relpath(root, iter_path)
# #             all_partitions.add(rel_partition)

# # print(f"Found {len(all_partitions)} unique partitions")

# # for partition in all_partitions:
# #     # find all parquet files for this partition in all iterations
# #     parquet_files = glob.glob(os.path.join(base_path, "iteration_*", partition, "*.parquet"))

# #     # Read and concatenate all parquet files into a single pandas DataFrame
# #     dfs = [pq.read_table(f).to_pandas() for f in parquet_files]
# #     merged_df = pd.concat(dfs, ignore_index=True)

# #     # Prepare output folder path
# #     out_folder = os.path.join(merged_base, partition)
# #     os.makedirs(out_folder, exist_ok=True)

# #     # Write merged parquet file (single file) for this partition
# #     out_file = os.path.join(out_folder, "part.0.parquet")
# #     pq.write_table(pa.Table.from_pandas(merged_df), out_file)

# #     print(f"Merged {len(parquet_files)} files for partition {partition} into {out_file}")

