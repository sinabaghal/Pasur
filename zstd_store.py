import zstandard as zstd
import torch, gzip, io, os
import psutil
import numpy as np 

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # in MB


def load_tensor(filepath):

    with open(filepath, "rb") as f:
        compressed = f.read()
    
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(compressed)
    
    buffer = io.BytesIO(decompressed)
    tensor = torch.load(buffer)
    
    return tensor



def load_numpy(filepath):

    with open(filepath, "rb") as f:
        compressed = f.read()
    
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(compressed)
    
    buffer = io.BytesIO(decompressed)
    array = np.load(buffer)
    # import pdb; pdb.set_trace()
    return array

# def load_tensor(filepath=None, memory_store=None, compress=False):

#     # return memory_store[filepath]
#     # mem_before = get_memory_usage_mb()

#     if compress:

#         compressed = memory_store[filepath]
#         dctx = zstd.ZstdDecompressor()
#         decompressed = dctx.decompress(compressed)
#         buffer = io.BytesIO(decompressed)
#         obj = torch.load(buffer)
    
#     else:

#         obj = memory_store[filepath]


#     # mem_after = get_memory_usage_mb()

#     # return obj, mem_after-mem_before
#     return obj



def to_memory(obj, folder, device = 'disk', memory_store=None, level=3):

    # memory_store[name] = obj.to(device)
    
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(buffer.getvalue())

    if device != 'disk':
            memory_store[folder] = compressed
    else:
        with open(folder, 'wb') as f:
            f.write(compressed)

    return memory_store 