import pandas as pd
import xgboost as xgb
import os 
import numpy as np 
os.makedirs(f"../MDL/", exist_ok=True)
def trainsinglexgboost(i):

    df = pd.read_parquet(f"../PRQ/D{i}")

    os.makedirs(f"../MDL/D{i}", exist_ok=True)
    X_t = df.drop(["SGM", "D"], axis=1)
    y_t = df["SGM"]

    for col in X_t.columns:
        X_t[col] = X_t[col].astype("category")

    for i_hnd in range(6):

        mask = X_t["H"] == i_hnd

        y = y_t[mask]
        X = X_t[mask]

        

        # Create DMatrix
        dtrain = xgb.DMatrix(data=X, label=y, enable_categorical=True)

        # Parameters
        params = {
            "objective": "reg:squarederror",  # or "binary:logistic", etc.
            "eval_metric": "rmse" ,           # print RMSE per round
            "max_depth": 16,
            "eta": 0.3,
            "tree_method": "gpu_hist",         # use GPU accelerated histogram algorithm
            "predictor": "gpu_predictor",    # optional: GPU for inference
            "subsample": 0.2,
            'colsample_bytree': 0.6
            }

        # Eval set
        evals = [(dtrain, "train")]
        evals_result = {}
        # Train model with verbose output
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=evals,
            evals_result = evals_result,
            verbose_eval=True  # prints every boosting round
        )
        y_pred = np.clip(model.predict(dtrain), 0, 1)
        max_abs_error = np.max(np.abs(y - y_pred))
        print(f"\n📈 {i_hnd}: {max_abs_error:.4f}")
        
    #     df_results = pd.DataFrame({
    #     "y_true": y,
    #     "y_pred": y_pred
    #     })

    #     df_results.to_csv(f"../MDL/D{i}/res{i}_{i_hnd}.csv")
    # # return y, y_pred
    # # Save model if needed
    #     model.save_model(f"../MDL/D{i}/model_{i}_{i_hnd}.xgb")


if __name__ == '__main__':

    trainsinglexgboost(0)
    # df_eval = pd.DataFrame(evals_result["train"])
    import pdb; pdb.set_trace()
