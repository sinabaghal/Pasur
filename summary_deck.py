import torch 
import pandas as pd 

l_k = [48,49,50,51]
l_q = [44,45,46,47]
l_j = [40, 41, 42, 43]
l_n = [x for x in range(40)]
l_c = range(0,52,4) 
l_p = [0,1,2,3,4,37,40,41,42,43]
l_s = [1,1,1,1,2,3,1,1,1,1]
l_o = [4,37]

t_dks = torch.load(f"decks_10000.pt")[:,4:]
N = t_dks.shape[0]
chunks = t_dks.view(N, 12, 4)
alx = chunks[:, ::2].reshape(N, -1)   # shape: [N, 24]
bob = chunks[:, 1::2].reshape(N, -1) # shape: [N, 24]

j_alx = torch.isin(alx, torch.tensor(l_j)).sum(dim=1)
j_bob = torch.isin(bob, torch.tensor(l_j)).sum(dim=1)
c_alx = torch.isin(alx, torch.tensor(l_c)).sum(dim=1)
c_bob = torch.isin(bob, torch.tensor(l_c)).sum(dim=1)
o_alx = torch.isin(alx, torch.tensor(l_o)).sum(dim=1)
o_bob = torch.isin(bob, torch.tensor(l_o)).sum(dim=1)

df = pd.DataFrame({
    "Deck": range(10000),
    "CA": c_alx,
    "JA": j_alx,
    "PA": o_alx,
    "CB": c_bob,
    "JB": j_bob,
    "PB": o_bob,
})


# j0_alx = torch.isin(alx[:,:4], torch.tensor(l_j)).sum(dim=1)
# j0_bob = torch.isin(bob[:,:4], torch.tensor(l_j)).sum(dim=1)
# j1_alx = torch.isin(alx[:,4:8], torch.tensor(l_j)).sum(dim=1)
# j1_bob = torch.isin(bob[:,4:8], torch.tensor(l_j)).sum(dim=1)
# j2_alx = torch.isin(alx[:,8:12], torch.tensor(l_j)).sum(dim=1)
# j2_bob = torch.isin(bob[:,8:12], torch.tensor(l_j)).sum(dim=1)
# j3_alx = torch.isin(alx[:,12:16], torch.tensor(l_j)).sum(dim=1)
# j3_bob = torch.isin(bob[:,12:16], torch.tensor(l_j)).sum(dim=1)
# j4_alx = torch.isin(alx[:,16:20], torch.tensor(l_j)).sum(dim=1)
# j4_bob = torch.isin(bob[:,16:20], torch.tensor(l_j)).sum(dim=1)
# j5_alx = torch.isin(alx[:,20:24], torch.tensor(l_j)).sum(dim=1)
# j5_bob = torch.isin(bob[:,20:24], torch.tensor(l_j)).sum(dim=1)

# c0_alx = torch.isin(alx[:,:4], torch.tensor(l_c)).sum(dim=1)
# c0_bob = torch.isin(bob[:,:4], torch.tensor(l_c)).sum(dim=1)
# c1_alx = torch.isin(alx[:,4:8], torch.tensor(l_c)).sum(dim=1)
# c1_bob = torch.isin(bob[:,4:8], torch.tensor(l_c)).sum(dim=1)
# c2_alx = torch.isin(alx[:,8:12], torch.tensor(l_c)).sum(dim=1)
# c2_bob = torch.isin(bob[:,8:12], torch.tensor(l_c)).sum(dim=1)
# c3_alx = torch.isin(alx[:,12:16], torch.tensor(l_c)).sum(dim=1)
# c3_bob = torch.isin(bob[:,12:16], torch.tensor(l_c)).sum(dim=1)
# c4_alx = torch.isin(alx[:,16:20], torch.tensor(l_c)).sum(dim=1)
# c4_bob = torch.isin(bob[:,16:20], torch.tensor(l_c)).sum(dim=1)
# c5_alx = torch.isin(alx[:,20:24], torch.tensor(l_c)).sum(dim=1)
# c5_bob = torch.isin(bob[:,20:24], torch.tensor(l_c)).sum(dim=1)


# o0_alx = torch.isin(alx[:,:4], torch.tensor(l_o)).sum(dim=1)
# o0_bob = torch.isin(bob[:,:4], torch.tensor(l_o)).sum(dim=1)
# o1_alx = torch.isin(alx[:,4:8], torch.tensor(l_o)).sum(dim=1)
# o1_bob = torch.isin(bob[:,4:8], torch.tensor(l_o)).sum(dim=1)
# o2_alx = torch.isin(alx[:,8:12], torch.tensor(l_o)).sum(dim=1)
# o2_bob = torch.isin(bob[:,8:12], torch.tensor(l_o)).sum(dim=1)
# o3_alx = torch.isin(alx[:,12:16], torch.tensor(l_o)).sum(dim=1)
# o3_bob = torch.isin(bob[:,12:16], torch.tensor(l_o)).sum(dim=1)
# o4_alx = torch.isin(alx[:,16:20], torch.tensor(l_o)).sum(dim=1)
# o4_bob = torch.isin(bob[:,16:20], torch.tensor(l_o)).sum(dim=1)
# o5_alx = torch.isin(alx[:,20:24], torch.tensor(l_o)).sum(dim=1)
# o5_bob = torch.isin(bob[:,20:24], torch.tensor(l_o)).sum(dim=1)


# df = pd.DataFrame({
#     "Deck": range(10000),

#     "J0_A": j0_alx, "J0_B": j0_bob,
#     "J1_A": j1_alx, "J1_B": j1_bob,
#     "J2_A": j2_alx, "J2_B": j2_bob,
#     "J3_A": j3_alx, "J3_B": j3_bob,
#     "J4_A": j4_alx, "J4_B": j4_bob,
#     "J5_A": j5_alx, "J5_B": j5_bob,
    
#     "C0_A": c0_alx, "C0_B": c0_bob,
#     "C1_A": c1_alx, "C1_B": c1_bob,
#     "C2_A": c2_alx, "C2_B": c2_bob,
#     "C3_A": c3_alx, "C3_B": c3_bob,
#     "C4_A": c4_alx, "C4_B": c4_bob,
#     "C5_A": c5_alx, "C5_B": c5_bob,
    
#     "P0_A": o0_alx, "P0_B": o0_bob,
#     "P1_A": o1_alx, "P1_B": o1_bob,
#     "P2_A": o2_alx, "P2_B": o2_bob,
#     "P3_A": o3_alx, "P3_B": o3_bob,
#     "P4_A": o4_alx, "P4_B": o4_bob,
#     "P5_A": o5_alx, "P5_B": o5_bob,
# })

import pdb; pdb.set_trace()