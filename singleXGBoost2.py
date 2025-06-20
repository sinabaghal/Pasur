import pandas as pd
import xgboost as xgb
import os 
import numpy as np 
from Imports import gammas, gamma_ids

os.makedirs(f"../MDL/", exist_ok=True)
csv_file = '../MDL/rmse_log.csv'

if not os.path.isfile(csv_file):
    pd.DataFrame(columns=['DECK', 'Gamma_ID', 'Iter', 'RMSE']).to_csv(csv_file, index=False)



def trainsinglexgboost(i_dck, id_gamma):

    gamma = gammas[id_gamma]
    df = pd.read_parquet(f"../PRQ/D{i_dck}")

    os.makedirs(f"../MDL/D{i_dck}", exist_ok=True)

    X = df.drop(["SGM", "D"], axis=1)
    y = df["SGM"]
    
    # mask = (df.H== 0)  & (df.P==1) & (df['T'] == 0) 
    # mask = (df.H==5) & (df.A==3) & (df.B == 6) & (df.W == 0) 
    # mask = df.H == 4
    # X = X[mask]
    # y = y[mask]
    # X = X.drop(["H"],axis=1)
    # X = X.drop(["A","B", "W", "H"],axis=1)
    
    

    # model = CatBoostRegressor(
    #     iterations=1000,
    #     learning_rate=0.05,
    #     depth=8,
    #     verbose=10,
    #     task_type="GPU")

    # model.fit(X, y, cat_features=list(X.columns))

        

    # Create DMatrix
    dtrain = xgb.DMatrix(data=X, label=y)

    # Parameters
    params = {
    "objective": "reg:squarederror",  # or "binary:logistic", etc.
    "eval_metric": "rmse" ,           # print RMSE per round
    "max_depth": 16,
    "eta": 0.1,
    "tree_method": "gpu_hist",         # use GPU accelerated histogram algorithm
    "predictor": "gpu_predictor",
    'lambda':0,
    'gamma':gamma
    # "gamma":0.01
    # optional: GPU for inference
    # "subsample": 0.2
    # 'colsample_bytree': 0.6
    }


    # Eval set
    evals = [(dtrain, "train")]
    evals_result = {}
    # Train model with verbose output
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=evals,
        evals_result = evals_result,
        verbose_eval=100, # prints every boosting round
        early_stopping_rounds=100
    )
    bins = np.arange(0, 1.05, 0.05)  # bins from 0 to 1 in steps of 0.05

    rmse_values = evals_result['train']['rmse']
    
    df = pd.DataFrame({
    'Deck': i_dck,
    'Gamma_ID': id_gamma,
    'Iter': range(len(rmse_values)),
    'RMSE': rmse_values
    })

    df.to_csv(csv_file, mode='a', header=False, index=False)
    y_pred = np.clip(model.predict(dtrain), 0, 1)
    # import pdb; pdb.set_trace()
    max_abs_error = np.abs(y - y_pred)
    hist, _ = np.histogram(max_abs_error, bins=bins)

    percentages = hist / len(max_abs_error) * 100

    for i in range(len(hist)):
        print(f"[{bins[i]:.2f}, {bins[i+1]:.2f}): {percentages[i]:.2f}%")


    # print(f"\n📈{max_abs_error:.4f}")
        
    # pd.DataFrame(y_pred).to_csv(f"../MDL/mdl_{i}.csv")

    #     df_results.to_csv(f"../MDL/D{i}/res{i}_{i_hnd}.csv")
    # # return y, y_pred
    # # Save model if needed
    model.save_model(f"../MDL/D{i_dck}/model_{i_dck}_{id_gamma}.xgb")


if __name__ == '__main__':

    N = 11
    # for i_dck in [f'{x}m' for x in range(N)]:
    for i_dck in [0]:
    # for i_dck in list(range(N))+[f'{x}m' for x in range(N)]:
        for id_gamma in [0]:
            trainsinglexgboost(i_dck, id_gamma)

    # df_eval = pd.DataFrame(evals_result["train"])
    
