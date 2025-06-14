import xgboost as xgb
from zstd_store import load_tensor
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.datasets import dump_svmlight_file
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gc 
# N = 1

# Xt_paths = [f"../DATA/nns/NNS_{i}.pt.zst" for i in range(N)]
# Yt_paths = [f"../DATA/SGM/SGM_{i}.pt.zst" for i in range(N)]
N =  2
XT, XV, YT, YV = [],[],[],[]

for i in range(N):

    X = load_tensor(f"../DATA/nns/NNS_{i}.pt.zst").numpy()
    Y = load_tensor(f"../DATA/SGM/SGM_{i}.pt.zst").numpy()

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    XT.append(X_train)
    YT.append(y_train)

    XV.append(X_val)
    YV.append(y_val)


XT = np.concatenate(XT, axis=0)
YT = np.concatenate(YT, axis=0)

XV = np.concatenate(XV, axis=0)
YV = np.concatenate(YV, axis=0)

# Xv_paths = [f"../DATA/nns/VNNS_{i}.pt.zst" for i in range(N)]
# Yv_paths = [f"../DATA/SGM/VSGM_{i}.pt.zst" for i in range(N)]

# X_val = np.concatenate([load_tensor(path).numpy() for path in Xv_paths], axis=0)
# Y_val = np.concatenate([load_tensor(path).numpy() for path in Yv_paths], axis=0)



dtrain = xgb.DMatrix(XT, label=YT)
dval = xgb.DMatrix(XV, label=YV)

del XT, YT, XV, YV, X, Y, X_train, X_val, y_train, y_val
gc.collect()

print("Data Ready!")
# print(X_train.nbytes / 1024**2)
# print(X_val.nbytes / 1024**2)
params = {
    "max_depth": 16,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "eta": 0.3

}

evallist = [(dtrain, 'train'), (dval, 'eval')]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evallist,
    early_stopping_rounds=50,
    verbose_eval=True
)

bst.save_model('../MODEL/model.json')
# bst = xgb.Booster()
# bst.load_model('model.json')
import pdb; pdb.set_trace()