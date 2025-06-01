import xgboost as xgb
from zstd_store import load_tensor
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

N = 2
bst = xgb.Booster()
bst.load_model('../MODEL/model.json')
for i in range(N):

    X = load_tensor(f"../DATA/nns/NNS_{i}.pt.zst").numpy()
    Y = load_tensor(f"../DATA/SGM/SGM_{i}.pt.zst").numpy() 

    dtest = xgb.DMatrix(X)

    del X
    y_pred = bst.predict(dtest)
    print(f"RMSE {i}:", np.sqrt(mean_squared_error(Y, y_pred)))
    print(f"MAE: {i}", mean_absolute_error(Y, y_pred))
    import pdb; pdb.set_trace()
