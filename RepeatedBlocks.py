import torch 
from Imports import device

def repeatedblocks(t_org, t_bsz, t_rpt):

    # Compute start t_idx of each block
    t_bgn = torch.repeat_interleave(torch.cumsum(torch.cat([torch.tensor([0], device=device), t_bsz[:-1]]), dim=0), t_rpt)

    # Create an index tensor for each element's original position
    # e.g., block 0 repeated 1 time, block 1 repeated 3 times, block 2 repeated 2 times
    # t_blk = torch.repeat_interleave(t_bgn, t_rpt)
    # tensor([0, 2, 2, 2, 5, 5])
    t_bls = torch.repeat_interleave(t_bsz, t_rpt)
    ## tensor([2, 3, 3, 3, 1, 1])

    ## identify each block using its first index
    t_blk = torch.repeat_interleave(t_bgn, t_bls) 
    ## b0,b0,b1,b1,b1,b1,b1,b1,b1,b1,b1,b2,b2
    ## tensor([0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5])

    ## shift starting value 
    ## tensor([ 0,  0,  2,  2,  2,  5,  5,  5,  8,  8,  8, 11, 12])
    t_csm = torch.cumsum(torch.cat([torch.tensor([0], device=device), t_bls[:-1]]), dim=0).repeat_interleave(t_bls)
    i_blk = t_blk.shape[0]
    t_pos = torch.arange(i_blk, device=device) - t_csm
    t_idx = t_pos+t_blk
    t_rbk = t_org[t_idx]
    return t_rbk


if __name__ == '__main__':

    t_org = torch.tensor([1, 2, 3, 4, 5, 6])      # 1D tensor
    t_bsz = torch.tensor([2, 3, 1])           # Block t_bsz: s₁, s₂, s₃
    t_rpt = torch.tensor([1, 3, 2])         # Repeat counts: a₁, a₂, a₃

    repeatedblocks(t_org,t_bsz,t_rpt) 

    # D_S2T = {s:tensor_from_string(s) for s in all_strings}
    # torch.save(D_S2T , 'D_S2T.pt')
    pass 
