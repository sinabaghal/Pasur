import torch 
from Imports import device, INT8, d_scr


def generate_valid_decks(N, device=device):

    jack_set = {40, 41, 42, 43}
    batch_size = N * 2  # Oversample for filtering

    while True:
        # Generate random noise and get permutations via argsort
        noise = torch.rand(batch_size, 52, device=device)
        decks = torch.argsort(noise, dim=1)

        # Check for jacks in first 4 cards
        first_four = decks[:, :4]
        has_jack = (first_four.unsqueeze(-1) == torch.tensor(list(jack_set), device=device)).any(dim=(1, 2))

        valid_decks = decks[~has_jack]

        if valid_decks.size(0) >= N:
            return valid_decks[:N]

        

def setup():



    t_dck = torch.tensor([47, 40, 45, 16, 23, 20,  1,  4, 21, 17, 36, 29, 18, 32, 49,  9, 34, 24,
        37, 41, 48, 44, 27,  8, 51,  2,  6,  0, 11, 10, 26, 33, 12, 42, 15, 13,
         3, 50, 39, 35, 43, 28,  7, 46, 30, 22, 19,  5, 25, 38, 31, 14],
       device='cuda:0')
    # t_dck = torch.tensor(range(52), device=device)
    # while True:

    #     t_dck = t_dck[torch.randperm(t_dck.size(0))]
    #     no_jack   = not any([x in [40, 41, 42, 43] for x in t_dck[:4] ])
    #     if no_jack:
    #         break 
        
    #     import pdb; pdb.set_trace()

    
    # to_zstd([t_dck], ['t_dck'], folder+f"Data",0,to_memory=True)

    t_m52 = torch.tensor([True if i in t_dck[:4] else False for i in range(52)], device=device)
    t_mdd = torch.tensor([False for _ in range(52)], device=device)
    t_inf = torch.zeros((1,3,4), dtype=INT8, device=device)
    t_inf[0,0,:] = 3
    clt_dck      = t_dck.clone()
    t_dck        = t_dck[4:]

    t_scr     = torch.zeros((1,len(d_scr)), device=device,dtype=INT8)
    t_sid = torch.zeros((1,2), device=device,dtype=torch.int32)

    return t_inf, t_sid, t_scr,t_dck, t_m52, t_mdd, clt_dck



