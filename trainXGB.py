import gc
import glob
import asyncio
import numpy as np
import xgboost as xgb
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from sklearn.metrics import mean_squared_error
from dask_ml.model_selection import train_test_split
import dask.config
from dask.distributed import Client, LocalCluster

# asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

if __name__ == '__main__':
   

    dask.config.set({
        'distributed.worker.memory.target': 0.5,
        'distributed.worker.memory.spill': 0.5,
        'distributed.worker.memory.pause': 0.95,
        'distributed.worker.memory.terminate': 0.98
        # 'distributed.worker.local-directory': '/mnt/d/Pasur/SPILL'
    })

    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        memory_limit='10GB'
    )

    client = Client(cluster)
    # import pdb; pdb.set_trace()
    # print(f"Dask dashboard: {client.dashboard_link}")
    files = glob.glob('../PRQ/0_200/5_3_0/*.parquet')
    # files = files[:10]
    df = dd.read_parquet(files)

    print(f"Data shape: {df.shape}")
    print(f"Number of partitions: {df.npartitions}")

    X = df.drop(['SGM','D'], axis=1)
    y = df['SGM']
    import pdb; pdb.set_trace()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    params = {
        'tree_method': 'hist',
        # 'gpu_id': 0,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'eta': 0.1,
        'n_estimators': 500,
        'subsample': 0.1,
        # Memory efficiency settings
        # 'max_bin': 256,  # Reduce memory usage
        # 'single_precision_histogram': True,
    }

    print("Starting out-of-core training...")

    model = xgb.dask.DaskXGBRegressor(
        **params,
        early_stopping_rounds=10,
        verbose_eval=10,
        verbose=True
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)]
    )

    print("Training completed!")

    print("Making predictions...")
    predictions = model.predict(X_test)

    y_test_computed = y_test.compute()
    predictions_computed = predictions.compute()
    mae = np.mean(np.abs(y_test_computed - predictions_computed))
    print(f"MAE: {mae:.4f}")

    # Save model
    model.save_model('xgb_out_of_core_model.json')
    print("Model saved!")

    # Cleanup
    del y_test_computed, predictions_computed
    gc.collect()

    client.close()