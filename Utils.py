import torch, math 
from collections import defaultdict
from Imports import device, INT8
import random 
import os

l_k = [48,49,50,51]
l_q = [44,45,46,47]
l_j = [40, 41, 42, 43]
l_n = [x for x in range(40)]
l_c = range(0,52,4) 
# l_c = [4*x for x in range(13)]
l_p = [0,1,2,3,4,37,40,41,42,43]
l_s = [1,1,1,1,2,3,1,1,1,1]

## Convert unique result to the full result for numeric cards 
def inverseunique(t_cnt,t_inx):

    t_cms = torch.zeros(t_cnt.shape[0]+1,dtype=torch.int32, device=device)
    t_cms[1:] = t_cnt.cumsum(0)
    # t_cms[1:] = torch.cumsum(t_cnt,dim=0)
    t_bgn, t_len = t_cms[t_inx], t_cnt[t_inx]
    # t_bgn, t_end = t_cms[t_inx], t_cms[t_inx + 1]
    # t_len = t_end - t_bgn  
    t_lns = t_len.cumsum(0)
    t_ofs = torch.arange(t_lns[-1], device=device) - torch.repeat_interleave(t_lns - t_len, t_len)
    # total_length = t_len.sum()  
    # t_inu = t_bgn.repeat_interleave(t_len) + t_ofs
    # assert torch.all(t_len == t_cnt[t_inx])
    return t_bgn.repeat_interleave(t_len) + t_ofs


# def inverseunique(t_cnt,t_inx):

#     t_cms = torch.zeros(t_cnt.shape[0]+1,dtype=torch.int32, device=device)
#     t_cms[1:] = torch.cumsum(t_cnt,dim=0)
#     t_bgn, t_len = t_cms[t_inx], t_cnt[t_inx]
#     t_lns = t_len.cumsum()
#     t_ofs = torch.arange(t_len.sum(), device=device) - torch.repeat_interleave(t_len.cumsum(0) - t_len, t_len)
#     t_inu = t_bgn.repeat_interleave(t_len) + t_ofs
#     return t_inu

def pad_helper(dm, cur):

    # i_n    = dm['cur_n'].sum()
    # i_j    = dm['cur_j'].sum()
    # i_q    = dm['cur_q'].sum()
    # i_k    = dm['cur_k'].sum()

    i_n, i_j, i_q, i_k = (dm[k].sum() for k in ['cur_n', 'cur_j', 'cur_q', 'cur_k'])
    # i_n, i_j, i_q, i_k = (t_msk.sum() for t_msk in [t_inn, t_inj, t_inq, t_ink])

    i_kq, i_jqk, i_njk, i_njq, i_njqk = i_k + i_q, i_j + i_q + i_k, i_n + i_j + i_k, i_n + i_j + i_q, i_n + i_j + i_q + i_k

    i_pad = {'n':  i_jqk, 'j': i_kq, 'k': i_njq, 'q': i_njk}
    t_pad = {'n':  None, 
             'j':  None, 
             'k': torch.hstack((torch.arange(i_k,i_njqk, device=device), torch.arange(i_k, device=device))), 
             'q': torch.hstack((torch.arange(i_q,i_njq, device=device), torch.arange(i_q, device=device),torch.arange(i_njq,i_njqk, device=device)))}
    
    return i_pad[cur], t_pad[cur]

    # i_njqk = i_n+i_j+i_q+i_k
    # i_njk = i_n+i_j+i_k
    # i_njq = i_n+i_j+i_q
    
    # if cur == 'j': 
    #     return i_k+i_q, None
    # elif cur == 'k':
    #     return i_njq, torch.hstack((torch.arange(i_k,i_njqk, device=device), torch.arange(i_k, device=device)))
    # elif cur == 'q':
    #     return i_njk, torch.hstack((torch.arange(i_q,i_njq, device=device), torch.arange(i_q, device=device),torch.arange(i_njq,i_njqk, device=device)))
    # else:
    #     return i_j + i_q + i_k, None

def partition(lst, size):
    random.shuffle(lst)
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def get_d_mask(tm_52, l_next_cards):
    # import pdb; pdb.set_trace()
    dm = defaultdict(list)
    t_nnzr = torch.nonzero(tm_52, as_tuple=True)[0]
    lm_nxt_a = l_next_cards[:4]
    lm_nxt_b = l_next_cards[4:8]
    tm_52[lm_nxt_a] = True
    tm_52[lm_nxt_b] = True 
    _, dm['pad'] = torch.sort(torch.cat([t_nnzr,l_next_cards[:8]]))
    dm['cur_s']  = torch.tensor(l_s,dtype = INT8, device=device)[tm_52[l_p]]
    dm['cur_52'] = tm_52


    find_dm = lambda lm :  torch.tensor([idx in lm for idx, item in enumerate(tm_52) if item], dtype = bool, device=device)

    dm['cur_k'] = find_dm(l_k)
    dm['cur_q'] = find_dm(l_q)
    dm['cur_j'] = find_dm(l_j)
    dm['cur_n'] = find_dm(l_n)
    dm['cur_c'] = find_dm(l_c)
    dm['cur_p'] = find_dm(l_p)
    dm['nxt_a'] = find_dm(lm_nxt_a)
    dm['nxt_b'] = find_dm(lm_nxt_b)


    return dm 




# Define folder and size limit
# folder = "save_folder/ADVT/nns"


def partition_folder(folder, MAX_FILES=100, MAX_TOTAL_SIZE_MB=100):
    
    size_limit_bytes = MAX_TOTAL_SIZE_MB * 1024 * 1024  # Convert MB to bytes

    # Get list of all files and their sizes
    files_with_sizes = [
        (f, os.path.getsize(os.path.join(folder, f)))
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]

    # Randomly shuffle and pick up to MAX_FILES
    random.shuffle(files_with_sizes)
    selected = files_with_sizes[:MAX_FILES]

    # Partition selected files into groups by size
    groups = []
    current_group = []
    current_size = 0

    for fname, fsize in selected:
        if current_size + fsize > size_limit_bytes:
            if current_group:
                groups.append(current_group)
            current_group = [(fname, fsize)]
            current_size = fsize
        else:
            current_group.append((fname, fsize))
            current_size += fsize

    if current_group:
        groups.append(current_group)

    # Convert filenames to full paths
    grouped_paths = [[os.path.join(folder, fname) for fname, _ in group] for group in groups]
    return grouped_paths




# label_width = 5
# clear = lambda: os.system('cls')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TYPE = torch.int8
# l_k = [48,49,50,51]
# l_q = [44,45,46,47]
# l_j = [40, 41, 42, 43]
# l_n = [x for x in range(40)]
# l_c = [4*x for x in range(13)]
# l_p = [0,1,2,3,4,37,40,41,42,43]
# l_s = [1,1,1,1,2,3,1,1,1,1]



# def ask_to_continue(p0_hand,p1_hand,pool,p0_clt,p1_clt):
    
#     input().strip().lower()
#     clear()
#     print("Pa:".ljust(label_width), *[card_unicode_map[card] for card in print_deck(p0_hand)])
#     print("Pb:".ljust(label_width), *[card_unicode_map[card] for card in print_deck(p1_hand)])
#     print("Po:".ljust(label_width), *[card_unicode_map[card] for card in print_deck(pool)])
#     print("Clt0:".ljust(label_width), *[card_unicode_map[card] for card in print_deck(torch.tensor(p0_clt))])
#     print("Clt1".ljust(label_width), *[card_unicode_map[card] for card in print_deck(torch.tensor(p1_clt))])




# def print_memory_usage(label=""):
#     allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
#     reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
#     print(f"{label} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


# def print_deck(deck):

#     # import pdb; pdb.set_trace()
#     ranks = [math.floor(t/4) for t in deck]
#     suits = [t.item()%4 for t in deck]
#     return [f"{RANKS[rank]}{SUITS[suit]}" for (rank,suit) in zip(ranks,suits)]


# def mm_check():

#     to_gb = lambda x : x/1024**3
#     total_memory = torch.cuda.memory_reserved()
#     allocated_memory = torch.cuda.memory_allocated()
#     free_memory = total_memory - allocated_memory  # Approximate free memory

#     print(f"Reserved Memory: {to_gb(total_memory):.2f} GB")
#     print(f"Allocated Memory: {to_gb(allocated_memory):.2f} GB")
#     print(f"Free Memory: {to_gb(free_memory):.2f} GB")


# def find_tz_deck(tm_52,row):
#     cur_deck = print_deck(torch.nonzero(tm_52))
#     test_deck = [cur_deck[i] for i in torch.nonzero(torch.tensor([(i>0).item() for i in row]),as_tuple=True)[0]] 
    
#     return test_deck



#     # dm['cur_k'] = torch.tensor([idx in l_k for idx, item in enumerate(tm_52) if item], dtype = bool, device=device)
#     # dm['cur_q'] = torch.tensor([idx in l_q for idx, item in enumerate(tm_52) if item], dtype = bool, device=device)
#     # dm['cur_j'] = torch.tensor([idx in l_j for idx, item in enumerate(tm_52) if item], dtype = bool, device=device)
#     # dm['cur_n'] = torch.tensor([idx in l_n for idx, item in enumerate(tm_52) if item], dtype = bool, device=device)
#     # dm['cur_c'] = torch.tensor([idx in l_c for idx, item in enumerate(tm_52) if item], dtype = bool, device=device)
#     # dm['cur_p'] = torch.tensor([idx in l_p for idx, item in enumerate(tm_52) if item], dtype = bool, device=device)
#     # dm['nxt_a'] = torch.tensor([idx in lm_nxt_a for idx, item in enumerate(tm_52) if item], dtype = bool, device=device)
#     # dm['nxt_b'] = torch.tensor([idx in lm_nxt_b for idx, item in enumerate(tm_52) if item], dtype = bool, device=device)

#     # for idx, item in enumerate(tm_52):
#     #     if item:
            
#     #         dm['cur_k'].append(idx in l_k) 
#     #         dm['cur_q'].append(idx in l_q) 
#     #         dm['cur_j'].append(idx in l_j) 
#     #         dm['cur_n'].append(idx in l_n) 
#     #         dm['cur_c'].append(idx in l_c) 
#     #         dm['cur_p'].append(idx in l_p)
#     #         dm['nxt_a'].append(idx in lm_nxt_a)
#     #         dm['nxt_b'].append(idx in lm_nxt_b)
            
#     # dm['cur_k']  = torch.tensor(dm['cur_k'], dtype = bool, device=device)
#     # dm['cur_q']  = torch.tensor(dm['cur_q'], dtype = bool, device=device)
#     # dm['cur_j']  = torch.tensor(dm['cur_j'], dtype = bool, device=device)
#     # dm['cur_c']  = torch.tensor(dm['cur_c'], dtype = bool, device=device)
#     # dm['cur_n']  = torch.tensor(dm['cur_n'], dtype = bool, device=device)
#     # dm['cur_p']  = torch.tensor(dm['cur_p'], dtype = bool, device=device)
#     # dm['nxt_a']  = torch.tensor(dm['nxt_a'], dtype = bool, device=device)
#     # dm['nxt_b']  = torch.tensor(dm['nxt_b'], dtype = bool, device=device)
#     return dm


# def abstraction(tm_52):

#     deck = print_deck(torch.tensor(range(52)))
#     strings = []
#     for idx, item in enumerate(deck):

#         if tm_52[idx]: 
#             if item[-1] == '♣':
                
#                 strings.append(f'{int(idx/4)}')
#             elif item == '10♦':
#                 strings.append('13')        
#             else:
#                 rank = RANKS.index(item[:-1])
#                 strings.append(f'{14+rank}')
    
    
#     index_map = {}
#     cur = 0

#     for string in strings:
#         if string not in index_map:
#             index_map[string] = cur
#             cur = cur+1
#     t_inv = dict(index_map)
#     return torch.tensor([t_inv[key] for key in strings], dtype=torch.int64, device=device)


# def display_row(t_row):

#     ## Print the corresponding card for each value 1 inside t_row
#     ranks = [math.floor(t.item()) for t in (t_row == 1).nonzero(as_tuple=False).flatten()/4]
#     suits = [t.item() for t in (t_row == 1).nonzero(as_tuple=False).flatten()%4]
#     return [f"{RANKS[rank]}{SUITS[suit]}" for (rank,suit) in zip(ranks,suits)]


# def pick_a_true(bool_tensor):

#     true_indices = torch.nonzero(bool_tensor, as_tuple=True)[0]  # Extract indices as a 1D tensor
#     random_index = true_indices[torch.randint(len(true_indices), (1,))].item()
#     return random_index

# def print_shapes(tensors):
#     for name, tensor in tensors.items():
#         print(f"{name}: {tensor.shape}")




# card_unicode_map = {
#     '1♠': '\U0001F0A1',  # Ace of Spades
#     '2♠': '\U0001F0A2',  # 2 of Spades
#     '3♠': '\U0001F0A3',
#     '4♠': '\U0001F0A4',
#     '5♠': '\U0001F0A5',
#     '6♠': '\U0001F0A6',
#     '7♠': '\U0001F0A7',
#     '8♠': '\U0001F0A8',
#     '9♠': '\U0001F0A9',
#     '10♠': '\U0001F0AA',
#     'J♠': '\U0001F0AB',  # Jack of Spades
#     'Q♠': '\U0001F0AD',  # Queen of Spades
#     'K♠': '\U0001F0AE',  # King of Spades

#     '1♥': '\U0001F0B1',  # Ace of Hearts
#     '2♥': '\U0001F0B2',
#     '3♥': '\U0001F0B3',
#     '4♥': '\U0001F0B4',
#     '5♥': '\U0001F0B5',
#     '6♥': '\U0001F0B6',
#     '7♥': '\U0001F0B7',
#     '8♥': '\U0001F0B8',
#     '9♥': '\U0001F0B9',
#     '10♥': '\U0001F0BA',
#     'J♥': '\U0001F0BB',
#     'Q♥': '\U0001F0BD',
#     'K♥': '\U0001F0BE',

#     '1♦': '\U0001F0C1',  # Ace of Diamonds
#     '2♦': '\U0001F0C2',
#     '3♦': '\U0001F0C3',
#     '4♦': '\U0001F0C4',
#     '5♦': '\U0001F0C5',
#     '6♦': '\U0001F0C6',
#     '7♦': '\U0001F0C7',
#     '8♦': '\U0001F0C8',
#     '9♦': '\U0001F0C9',
#     '10♦': '\U0001F0CA',
#     'J♦': '\U0001F0CB',
#     'Q♦': '\U0001F0CD',
#     'K♦': '\U0001F0CE',

#     '1♣': '\U0001F0D1',  # Ace of Clubs
#     '2♣': '\U0001F0D2',
#     '3♣': '\U0001F0D3',
#     '4♣': '\U0001F0D4',
#     '5♣': '\U0001F0D5',
#     '6♣': '\U0001F0D6',
#     '7♣': '\U0001F0D7',
#     '8♣': '\U0001F0D8',
#     '9♣': '\U0001F0D9',
#     '10♣': '\U0001F0DA',
#     'J♣': '\U0001F0DB',
#     'Q♣': '\U0001F0DD',  # Queen of Clubs
#     'K♣': '\U0001F0DE'
# }

# def tensor_from_string(s):
    
#     pos1 = [i for i, char in enumerate(s) if char == '1']
#     pos2 = [i for i, char in enumerate(s) if char == '2']

#     t_pairs = torch.tensor(list(itertools.product(pos1, pos2)))
#     t = torch.zeros((t_pairs.shape[0], 2,5), device=device, dtype=TYPE)
#     t_arange = torch.arange(t.shape[0], device=device)

#     if t.shape[0] == 0 : return t[:,:,:-1]

#     t[t_arange, 0, t_pairs[:, 0]] = 1
#     t[t_arange, 1, t_pairs[:, 1]] = 1
#     return t[:,:,:-1]

# all_strings = [
#     ''.join(map(str, p)) + '2' if 2 not in p else ''.join(map(str, p))+'0'
#     for p in itertools.product([0, 1, 2], repeat=4)
# ]


# import zipfile
# import os
# from pathlib import Path
# import shutil

# # Define paths
# unzipped_dir = Path("Unzipped")
# zipped_dir = Path("Zipped")

# def unzip_to_unzipped(folder_name):
#     """
#     Unzips 'folder_name.zip' from ZippedData into UnzippedData/folder_name
#     """
#     zip_path = zipped_dir / f"{folder_name}.zip"
#     extract_path = unzipped_dir / folder_name

#     if not zip_path.exists():
#         print(f"Zip file {zip_path} not found.")
#         return

#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_path)
    
#     print(f"Unzipped '{folder_name}.zip' to '{extract_path}'")

# def remove_from_unzipped(folder_name):
#     """
#     Removes UnzippedData/folder_name
#     """
#     folder_path = unzipped_dir / folder_name

#     if folder_path.exists() and folder_path.is_dir():
#         shutil.rmtree(folder_path)
#         print(f"Removed folder: {folder_path}")
#     else:
#         print(f"Folder {folder_path} not found.")



# def setup_folder(folder_path):
#     # If folder exists, remove it entirely
#     if os.path.exists(folder_path):
#         shutil.rmtree(folder_path, ignore_errors=True)  # Remove folder and all contents
#     # Create fresh folder
#     os.makedirs(folder_path)
#     # print(f"Folder ready: {folder_path}")





