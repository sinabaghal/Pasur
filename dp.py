import torch, math 

DP_TYPE = torch.float16
TARGET_SUM = 11
N = 40 # 40 cards in total 
VALS = torch.tensor([math.floor(1 + i/4) for i in range(40)], dtype=torch.int64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dp = torch.zeros((N + 1, TARGET_SUM + 1), dtype=torch.bool)
dp[0, 0] = True
for i in range(1, N + 1):
    v = VALS[i - 1].item()  
    for s in range(TARGET_SUM + 1):
        if dp[i - 1, s]:
            dp[i, s] = True
        elif s >= v and dp[i - 1, s - v]:
            dp[i, s] = True

    def backtrack(i, s):
        if s == 0:
            yield []
            return

        if i == 0: return

        if dp[i - 1, s]:
            for subset in backtrack(i - 1, s):
                yield subset

        v = VALS[i - 1].item()
        if s >= v and dp[i - 1, s - v]:
            for subset in backtrack(i - 1, s - v):
                yield subset + [i - 1]

    slns = []
    for subset_indices in backtrack(N, TARGET_SUM):
        sln_tensor = torch.zeros(N, dtype=DP_TYPE, device=device)
        for idx in subset_indices:
            sln_tensor[idx] = 1
        slns.append(sln_tensor)

## In total, there are 2764 different combinations of the 40 cards with sum equal to 11.
t_40x2764 = torch.stack(slns, dim=1) # Shape: [40, 2764]
t_2764x40 = t_40x2764.transpose(0,1) # Shape: [40, 2764]
slns_num_cards = t_40x2764.sum(dim=0) #[2764]
t_tuples = torch.stack((torch.ones(slns_num_cards.shape[0],dtype=DP_TYPE, device=device),slns_num_cards - 1),dim=1).transpose(0,1) #[2, 2764]