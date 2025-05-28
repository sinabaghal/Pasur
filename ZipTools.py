import zstandard as zstd
import torch, gzip, io
import os
from Imports import device
# s_fld = lambda seed: f'..\\Deck\\deck_{seed}\\'
from zstd_store import memory_store

def to_zstd(objs, names, folder, level=3, to_memory=False):

    for obj, name in zip(objs, names):
        filepath = f'{folder}/{name}.zstd'

        buffer = io.BytesIO()
        torch.save(obj.to('cpu'), buffer)
        cctx = zstd.ZstdCompressor(level=level)
        compressed = cctx.compress(buffer.getvalue())

        if to_memory:
            memory_store[filepath] = compressed
        else:
            os.makedirs(folder, exist_ok=True)
            with open(filepath, 'wb') as f:
                f.write(compressed)

    return memory_store if to_memory else None


def from_cpu_nn(i_trv, t, i_cfr, i_ext, nn_dtype, memory_store, ann):

    filepath = f'../STRG/CFR_{i_cfr}/EXT_{i_ext}/TRV_{i_trv}/AdvanMemory/t_nn.zstd'
    tensor = load_zstd(filepath, from_memory=True, memory_store=memory_store).to(dtype=nn_dtype)
    factor = torch.sqrt(torch.tensor(t + 1, dtype=nn_dtype))
    return factor * ann(tensor)

def from_cpu(i_trv, t, i_cfr, i_ext, nn_dtype, memory_store, to_memory, folder = 'AdvanMemory', tensor='t_reg'):

    filepath=f'../STRG/CFR_{i_cfr}/EXT_{i_ext}/TRV_{i_trv}/{folder}/{tensor}.zstd'
    tensor = load_zstd(filepath, from_memory=to_memory, memory_store=memory_store)
    factor = torch.sqrt(torch.tensor(t + 1, dtype=nn_dtype))
    return factor * tensor

def load_zstd(filepath=None, from_memory=False, memory_store=None):
    if from_memory:
        if memory_store is None or filepath not in memory_store:
            raise ValueError("Missing or invalid memory_store or filepath not found.")
        compressed = memory_store[filepath]
    else:
        with open(filepath, 'rb') as f:
            compressed = f.read()

    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(compressed)
    buffer = io.BytesIO(decompressed)
    obj = torch.load(buffer)
    return obj


def from_zstd(folder):

    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    d_fls = {f.split('.')[0]: load_zstd(folder+f) for f in files}

    return d_fls

def to_gz(filepath):
    filepath = filepath+'.gz'
    with open(filepath, 'rb') as f:
        decompressed_data = gzip.decompress(f.read())
    return torch.load(io.BytesIO(decompressed_data))



def to_gz(obj,id,compresslevel=3):
    
    filepath = filepath+'.gz'
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    compressed_data = gzip.compress(buffer.getvalue(), compresslevel=compresslevel)
    with open(filepath, 'wb') as f:
        f.write(compressed_data)



# def get_var_name(var):
#     frame = inspect.currentframe().f_back
#     for name, val in frame.f_locals.items():
#         if val is var:
#             return name
#     return None