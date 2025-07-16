
import torch 
import numpy as np 

INT8 = torch.int8
INT32 = torch.int32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gammas    = np.append(np.linspace(0,1,10000), [2,5,10])
# gamma_ids = [0,9999,9999+3] 
gamma_ids = [0]
# gamma_ids = [0,1,10,100,1000,9999,9999+1,9999+2,9999+3] 
d_scr = {'a_clb':0, 'b_clb':1, 'pts_dlt':2, 'max_clb':3}
d_rus = {'a_clb':0,'b_clb':1,'pts_dlt':2, 'lst_pck':3,'a_pts':4,'a_sur':5,'b_pts':6,'b_sur':7}


t_40x2764   = torch.load('helper_pts/t_40x2764.pt',map_location=device)
t_2764x40   = torch.load('helper_pts/t_2764x40.pt',map_location=device)
t_tpl    = torch.load('helper_pts/t_tuples.pt' ,map_location=device)
D_S2T       = torch.load('helper_pts/D_S2T.pt'    , map_location=device)