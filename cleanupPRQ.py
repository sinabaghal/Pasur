import os 
import torch 
import glob
from zstd_store import load_tensor
import pandas as pd 
import pyarrow as pa
import pyarrow.parquet as pq
import dask.dataframe as dd
import numpy as np 


folders = glob.glob("../PRQ/**/*.parquet", recursive=True)
ddf = dd.read_parquet(folders, engine="pyarrow").repartition(partition_size="1G")
for delayed_partition in ddf.to_delayed():
    partition = delayed_partition.compute()  # This is now a Pandas DataFrame
    import pdb; pdb.set_trace()