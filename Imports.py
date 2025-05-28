
import torch 

INT8 = torch.int8
INT32 = torch.int32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_scr = {'a_clb':0, 'b_clb':1, 'pts_dlt':2, 'max_clb':3}
d_snw = {'a_clb':0,'b_clb':1,'pts_dlt':2, 'lst_pck':3,'a_pts':4,'a_sur':5,'b_pts':6,'b_sur':7}


t_40x2764   = torch.load('helper_pts/t_40x2764.pt',map_location=device)
t_2764x40   = torch.load('helper_pts/t_2764x40.pt',map_location=device)
t_tuples    = torch.load('helper_pts/t_tuples.pt' ,map_location=device)
D_S2T       = torch.load('helper_pts/D_S2T.pt'    , map_location=device)