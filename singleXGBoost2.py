import pandas as pd
import xgboost as xgb
import os 
import numpy as np 
from catboost import CatBoostRegressor
os.makedirs(f"../MDL/", exist_ok=True)
def trainsinglexgboost(i_dck, gamma):

    df = pd.read_parquet(f"../PRQ/D{i_dck}")

    os.makedirs(f"../MDL/D{i_dck}", exist_ok=True)
    X = df.drop(["SGM", "D"], axis=1)
    y = df["SGM"]
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

    y_pred = np.clip(model.predict(dtrain), 0, 1)
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
    model.save_model(f"../MDL/D{i_dck}/model_{i_dck}_{gamma}.xgb")


if __name__ == '__main__':

    for gamma in np.array([1000]):

        trainsinglexgboost(8, gamma)
    # df_eval = pd.DataFrame(evals_result["train"])
    
