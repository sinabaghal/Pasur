import torch, math 
from SelfPlay import playrandom
# from NeuralNet import SNN
from Imports import d_rus, d_scr
import torch, sys, os, math
from Imports import device, INT8, INT32, d_rus, d_scr
# from NeuralNet import SNN, init_weights_zero
# import torch.nn as nn
import subprocess, re
# from PyPDF2 import PdfMerger  
import xgboost as xgb
import numpy as np 
# from tqdm import trange, tqdm
# import numpy as np 


RANKS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
SUITS = ['♣', '♦', '♥', '♠'] 
folder = 'games'

# Regex pattern: matches 'game_' followed by digits and ending with '.txt'
pattern = re.compile(r'^game_\d+\.txt$')

# Loop through files in the folder
for filename in os.listdir(folder):
    if pattern.match(filename):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)

latex_0 = r"""

\documentclass[a4paper, 12pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{array}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{graphicx} % For \resizebox
\input{cardscommands}
\usepackage{xcolor}
\usepackage{amssymb}
\usepackage{caption}
\usepackage{colortbl}
\arrayrulecolor{black} 
\usepackage{float} % for [H] placement, optional
\definecolor{myrowcolor}{HTML}{E0E0E0}  % Light gray
\begin{document}
\begin{figure}[H]  % or [htbp] if you want it to float
\begin{center}
\resizebox{1\textwidth}{!}{%
\begin{tabular}{|>{\centering\arraybackslash}p{0.1\textwidth}|
                >{\centering\arraybackslash}p{0.2\textwidth}|
                >{\centering\arraybackslash}p{0.2\textwidth}|
                >{\centering\arraybackslash}p{0.7\textwidth}|
                >{\centering\arraybackslash}p{0.05\textwidth}|
                 >{\centering\arraybackslash}p{0.7\textwidth}|
                >{\centering\arraybackslash}p{0.05\textwidth}|
                >{\centering\arraybackslash}p{0.05\textwidth}|
                 >{\centering\arraybackslash}p{0.05\textwidth}|
                  >{\centering\arraybackslash}p{0.05\textwidth}|
                   >{\centering\arraybackslash}p{0.05\textwidth}|
                   >{\centering\arraybackslash}p{0.05\textwidth}|
                   >{\centering\arraybackslash}p{0.05\textwidth}|
                   >{\centering\arraybackslash}p{0.05\textwidth}|}
\hline
\textbf{Stage}& \textbf{Alex} & \textbf{Bob}& \textbf{Pool} & \textbf{Lay} & \textbf{Pick} & \textbf{Acl} &  \textbf{Bcl} & \textbf{Apt} &  \textbf{Bpt} & \textbf{Asr} & \textbf{Bsr} & \textbf{$\Delta$} & \textbf{L}\\
\hline
\rowcolor{myrowcolor}
"""

latex_1 = lambda game_id, score: r"""\end{tabular}
}
\end{center}
\caption{Score = """ + str(score) + r"""}
\label{fig:game_""" + str(game_id) + r"""}
\end{figure}

\end{document}
"""
card_to_latex = {
    # Hearts (♥, red)
    '2♥': r'\TwH',
    '3♥': r'\ThH',
    '4♥': r'\FoH',
    '5♥': r'\FiH',
    '6♥': r'\SiH',
    '7♥': r'\SeH',
    '8♥': r'\EH',
    '9♥': r'\NH',
    '10♥': r'\TeH',
    'J♥': r'\JH',
    'Q♥': r'\QH',
    'K♥': r'\KH',
    '1♥': r'\AH',

    # Diamonds (♦, red)
    '2♦': r'\TwD',
    '3♦': r'\ThD',
    '4♦': r'\FoD',
    '5♦': r'\FiD',
    '6♦': r'\SiD',
    '7♦': r'\SeD',
    '8♦': r'\ED',
    '9♦': r'\ND',
    '10♦': r'\TeD',
    'J♦': r'\JD',
    'Q♦': r'\QD',
    'K♦': r'\KD',
    '1♦': r'\AD',

    # Clubs (♣, black)
    '2♣': r'\TwC',
    '3♣': r'\ThC',
    '4♣': r'\FoC',
    '5♣': r'\FiC',
    '6♣': r'\SiC',
    '7♣': r'\SeC',
    '8♣': r'\EC',
    '9♣': r'\NC',
    '10♣': r'\TeC',
    'J♣': r'\JC',
    'Q♣': r'\QC',
    'K♣': r'\KC',
    '1♣': r'\AC',

    # Spades (♠, black)
    '2♠': r'\TwS',
    '3♠': r'\ThS',
    '4♠': r'\FoS',
    '5♠': r'\FiS',
    '6♠': r'\SiS',
    '7♠': r'\SeS',
    '8♠': r'\ES',
    '9♠': r'\NS',
    '10♠': r'\TeS',
    'J♠': r'\JS',
    'Q♠': r'\QS',
    'K♠': r'\KS',
    '1♠': r'\AS',
}

def tp(t_row, i_ind):

    ## Print the corresponding card for each value 1 inside t_row
    ranks = [math.floor(t.item()) for t in (t_row == i_ind).nonzero(as_tuple=False).flatten()/4]
    suits = [t.item() for t in (t_row == i_ind).nonzero(as_tuple=False).flatten()%4]
    res   = [f"{RANKS[rank]}{SUITS[suit]}" for (rank,suit) in zip(ranks,suits)]
    return res 

def tp2(res, f):

    if len(res) == 0:
        print('&', end='',file=f)
        return 


    print('$', end='',file=f)
    for i, x in enumerate(res):
        end_char = r'\;' if i < len(res) - 1 else ''
        print(card_to_latex[x] + end_char, end='', file=f)
    print('$&', end='',file=f)
   
def find_score(t):

    res = 0
    if t[-1] == 1:
        res += 7
    else:
        res -= 7

    return (res+t[2]).item()

if __name__ == "__main__":

    N = 100
    i_dcks = [10]
    t_dks = torch.load(f"decks_10000.pt")

    wins = {'RR':[],'MR':[], 'RM':[], 'MM':[]}
    scores = {'RR':[],'MR':[], 'RM':[], 'MM':[]}
    
    for i_dck in range(152,153):

        # bst = xgb.Booster()
        # bst.load_model(f'../MDL/D{i_dck}/model_{i_dck}_cc.xgb')
        
        t_dck = t_dks[i_dck,:].to(device)
        # res = torch.tensor([((x - 4) // 4) % 2 if x >= 4 else 3 for x in range(52)])
        
       

        models = [('R','random')]
        # models = [('R','random'), ('M',bst)]
        for mdl0 in models:
            for mdl1 in models:

                code = f'{mdl0[0]}{mdl1[0]}'
                t_fsc, t_scr_, t_ltx_ = playrandom(t_dck, N=N, x_alx = mdl0[1] , x_bob = mdl1[1], to_latex = True)
                win   = 100*(t_fsc >= 0).sum().item()/N
                score = t_fsc.sum().item() / N 
                wins[code].append(win)
                scores[code].append(score)
                print(f'{code}{i_dck:<3} | Alex wins = {100*(t_fsc >= 0).sum().item() / N:6.2f}% | Average Score = {t_fsc.sum().item() / N:7.2f}')

    # # import pdb; pdb.set_trace()
    # print(100*"**")
    # for key in wins.keys():
    #     # import pdb; pdb.set_trace()
    #     print(f'{key}: Average Win Rate = {np.array(wins[key]).mean():6.2f} | Average Score = {np.array(scores[key]).mean():7.2f}')

        
    # import pdb; pdb.set_trace()
    # wins = np.array(wins)
    # scores = np.array(scores)
    # print(f'Average Win Rate = {wins.mean():6.2f} | Average Score = {scores.mean():7.2f}')

    # import pdb; pdb.set_trace()
        # break 
        # import pdb; pdb.set_trace()
        # a2 = torch.sum(t_fsc2)
        # import pdb; pdb.set_trace()
        # t_fsc3, t_scr_, t_ltx_ = playrandom(t_dck, N=N, x_alx = 'random' , x_bob = 'random', to_latex = True)
        # a3 = torch.sum(t_fsc3)
        
        # t_fsc4, t_scr_, t_ltx_ = playrandom(t_dck, N=N, x_alx = bst , x_bob = bst, to_latex = True)
        # a4 = torch.sum(t_fsc4)

        # a= torch.max(torch.tensor([a1,a2,a3,a4]))
        # import pdb; pdb.set_trace()
        # i_gin = 0

        # with tqdm(seeds, desc="SelfPlaying:") as pbar:

        #     for seed in pbar:

                
        #         # t0_win, _ = selfplay(seed=seed, N=1000, to_latex=True, x_alx=booster,  x_bob=xrnn_bob)
        #         # t1_win, _ = selfplay(seed=seed, N=10000, to_latex=False, x_alx=xrnn_alx, x_bob=xrnn_bob)

        #         # i_gin += (t0_win - t1_win).item()
        #         i_gin   += t0_win.item() 
        #         pbar.set_postfix(gain=f"{i_gin / (pbar.n + 1):.4f}")


        # i_inp = 4*52+4
        # layer_dims = [i_inp, 64, 32,1]
        # snn_alx = SNN(layer_dims, activation=nn.Sigmoid, dropout_p=0.3).to(device).apply(init_weights_zero)
        # snn_bob = SNN(layer_dims, activation=nn.Sigmoid, dropout_p=0.3).to(device).apply(init_weights_zero)
        # N = 1000
        # t_win, t_ltx_ = selfplay(seed=10, N=N, to_latex = True, snn_alx = snn_alx,snn_bob=snn_bob)


        for i_smp in range(N):

            # if i_smp != 111:
            #     continue
            # print(i_smp)
            # import pdb; pdb.set_trace()
            t_ltx = {key:t_ltx_[key][i_smp,:] for key in t_ltx_.keys()}
        
            with open(f"games/game_{i_smp}.txt", "w") as f:

                print(latex_0, file=f)

                for i_hnd in range(6):
                    
                    t_scr = t_ltx[f't_scr_{i_hnd}']

                    for i_trn in range(4):
                        for i_ply in range(2):

                            i_cod   = f'{i_hnd}_{i_trn}_{i_ply}'
                            print(f'\n {i_hnd}\_{i_trn}\_{i_ply} &', file=f, end='')
                            
                            clt_inf = t_ltx['t_dck_'+i_cod]
                            clt_act = t_ltx['t_act_'+i_cod] 
                            t_snw   = t_ltx['t_snw_'+i_cod] 
                            # print('t_snw_'+i_cod)
                            # print(t_snw)

                            alx     =  tp(clt_inf,1)
                            bob     =  tp(clt_inf,2)
                            pol     =  tp(clt_inf,3)
                            lay     =  tp(clt_act[0],1)
                            pck     =  tp(clt_act[1],1)
                            dlt     =  t_scr[d_scr['pts_dlt']]
                            lst     =  t_snw[d_rus['lst_pck']]
                            # if f'{i_trn}_{i_ply}' == '3_1':
                            #     import pdb; pdb.set_trace()
                            if lst == 1:   lst = 'A'
                            elif lst == 2: lst = 'B'
                            elif lst == 0:
                                pass
                            else:
                                NotImplementedError('ERROR')
                            
                            tp2(alx,f)
                            tp2(bob,f)
                            tp2(pol,f)
                            tp2(lay,f)
                            tp2(pck,f)

                            # import pdb; pdb.set_trace()
                            # \textbf{Acl} &  \textbf{Bcl} & \textbf{Apt} &  \textbf{Bpt} & \textbf{Asr} & \textbf{Bsr} & \textbf{$\Delta$}\\
                            a_clb, a_sur, a_pt = t_snw[d_rus['a_clb']], t_snw[d_rus['a_sur']], t_snw[d_rus['a_pts']] 
                            b_clb, b_sur, b_pt = t_snw[d_rus['b_clb']], t_snw[d_rus['b_sur']], t_snw[d_rus['b_pts']] 

                            a_clb += t_scr[d_scr['a_clb']]
                            b_clb += t_scr[d_scr['b_clb']]
                            
                            print(f'{a_clb} &', end='',file=f)
                            print(f'{b_clb} &', end='',file=f)
                            print(f'{a_pt} &', end='',file=f)
                            print(f'{b_pt} &', end='',file=f)
                            print(f'{a_sur} &', end='',file=f)
                            print(f'{b_sur} &', end='',file=f)
                            print(f'{dlt} &', end='',file=f)
                            print(f'{lst}', end='',file=f)
                            # if f'{i_hnd}_{i_trn}_{i_ply}' == '3_3_0':
                            #     import pdb; pdb.set_trace()
                            # print(f'{i_cod}, {a_sur},{b_sur} &')
                            # print(f'{i_cod}, {b_sur} &')
                            # if a_sur+b_sur> 0:
                            #     import pdb; pdb.set_trace()
                            print(r"\\ \hline", file=f)
                            if i_ply == 1:
                                if i_hnd == 5 and i_trn ==3:
                                    pass 
                                else:
                                    print(r" \rowcolor{myrowcolor}", file=f)

                            
                            
                        

                    print("\hline \n", file=f)
                    # break 
                print(f'CleanUp &', file=f, end='')
                tp2(torch.empty(0),f)
                tp2(torch.empty(0),f)
                tp2(torch.empty(0),f)
                tp2(torch.empty(0),f)
                tp2(torch.empty(0),f)
                t_snw  = t_ltx['t_snw_6'] 
                a_clb, a_sur, a_pt = t_snw[d_rus['a_clb']], t_snw[d_rus['a_sur']], t_snw[d_rus['a_pts']] 
                b_clb, b_sur, b_pt = t_snw[d_rus['b_clb']], t_snw[d_rus['b_sur']], t_snw[d_rus['b_pts']] 

                a_clb += t_scr[d_scr['a_clb']]
                b_clb += t_scr[d_scr['b_clb']]
                
                print(f'{a_clb} &', end='',file=f)
                print(f'{b_clb} &', end='',file=f)
                print(f'{a_pt} &', end='',file=f)
                print(f'{b_pt} &', end='',file=f)
                print(f'{a_sur} &', end='',file=f)
                print(f'{b_sur} &', end='',file=f)
                print(f'{dlt} &', end='',file=f)
                print(f'{lst}', end='',file=f)
                
                print(r"\\ \hline", file=f)
                print(latex_1(i_smp, find_score(t_scr_[i_smp,:])), file=f)



        import pdb; pdb.set_trace()