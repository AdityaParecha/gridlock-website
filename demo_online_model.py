# demo_online_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from river import forest, preprocessing, compose

DEMO_DIR = Path("demo_data")
DEMO_DIR.mkdir(exist_ok=True)
DEMO_BUFFER_PATH = DEMO_DIR / "demo_buffer.csv"
DEMO_MODEL_PATH = DEMO_DIR / "demo_online_model.pkl"
DEMO_RETRAIN_THRESHOLD = 50  # number of new records before the online model "updates" its view

def _load_or_init_model():
    if DEMO_MODEL_PATH.exists():
        import pickle
        with open(DEMO_MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    # Online random forest regressor, incrementally trainable one row at a time
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        forest.ARFRegressor(seed=42)
    )
    return model

def _save_model(model):
    import pickle
    with open(DEMO_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

def add_demo_record(lat, lon, target_time):
    # Append to buffer (demo-only file, never touches production data)
    row = pd.DataFrame([{
        'latitude': lat, 'longitude': lon,
        'created_datetime': target_time,
        'hour': target_time.hour, 'day_of_week': target_time.dayofweek
    }])
    if DEMO_BUFFER_PATH.exists():
        row.to_csv(DEMO_BUFFER_PATH, mode='a', header=False, index=False)
    else:
        row.to_csv(DEMO_BUFFER_PATH, index=False)

    buffer_df = pd.read_csv(DEMO_BUFFER_PATH)
    n_records = len(buffer_df)

    model = _load_or_init_model()

    # Online learning: update model incrementally with this single new point
    # Target = "density score" proxy: here we just teach it to predict lat/lon clustering tendency
    # given hour/day — replace with your real demo objective as needed.
    x = {'hour': target_time.hour, 'day_of_week': target_time.dayofweek, 'lat': lat, 'lon': lon}
    y = 1.0  # one observed violation event
    model.learn_one(x, y)
    _save_model(model)

    remaining = DEMO_RETRAIN_THRESHOLD - (n_records % DEMO_RETRAIN_THRESHOLD)
    return (f"Recorded. Demo model updated incrementally (no full retrain). "
            f"{n_records} total demo records so far. "
            f"{remaining} more until next milestone snapshot.")

def get_demo_stats():
    if not DEMO_BUFFER_PATH.exists():
        return {"n_records": 0}
    buffer_df = pd.read_csv(DEMO_BUFFER_PATH)
    return {"n_records": len(buffer_df)}