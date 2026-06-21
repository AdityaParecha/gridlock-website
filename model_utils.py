import pandas as pd
import numpy as np
import joblib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "knn_zone_dbscan2.pkl"
ZONED_DATA_PATH = BASE_DIR / "data" / "zoned_data.csv"
CENTROIDS_PATH = BASE_DIR / "zone_centroids.csv"

FEATURES = [
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
    'lag_1_step', 'lag_2_step', 'lag_4_step', 'lag_48_step',
    'roll_mean_4', 'roll_std_4', 'zone_hist_mean'
]

_model = None
_history_df = None
_centroids_df = None

# In model_utils.py, inside load_resources():
def load_resources():
    global _model, _history_df, _centroids_df
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _history_df is None:
        df = pd.read_csv(ZONED_DATA_PATH)
        df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce', utc=True)
        df = df.dropna(subset=['created_datetime'])
        df['zone_dbscan'] = df['zone_dbscan'].astype(str)          # NEW: force consistent dtype
        _history_df = df.set_index('created_datetime').groupby('zone_dbscan').resample('30min').size().reset_index(name='violation_count')
        _history_df = _history_df.sort_values(by=['zone_dbscan', 'created_datetime'])

        # NEW: pre-split into a dict once, instead of filtering the full df per zone per request
        global _zone_series_cache
        _zone_series_cache = {
            z: g.sort_values('created_datetime')
            for z, g in _history_df.groupby('zone_dbscan')
        }
    if _centroids_df is None:
        _centroids_df = pd.read_csv(CENTROIDS_PATH)
        _centroids_df['zone_dbscan'] = _centroids_df['zone_dbscan'].astype(str)  # NEW: force consistent dtype
    return _model, _history_df, _centroids_df

_zone_series_cache = None

def _engineer_features_for_target(zone_id, target_time, zone_series_cache):
    zone_hist = zone_series_cache.get(str(zone_id))
    if zone_hist is None:
        return None
    zone_hist = zone_hist[zone_hist['created_datetime'] < target_time]
    if zone_hist.empty:
        return None

    counts = zone_hist['violation_count'].values
    lag_1 = counts[-1] if len(counts) >= 1 else 0
    lag_2 = counts[-2] if len(counts) >= 2 else 0
    lag_4 = counts[-4] if len(counts) >= 4 else 0
    lag_48 = counts[-48] if len(counts) >= 48 else 0
    roll_mean_4 = counts[-4:].mean() if len(counts) >= 4 else counts.mean()
    roll_std_4 = counts[-4:].std() if len(counts) >= 4 else 0
    zone_hist_mean = counts.mean()

    hour = target_time.hour
    dow = target_time.dayofweek

    return {
        'hour_sin': np.sin(2 * np.pi * hour / 24.0),
        'hour_cos': np.cos(2 * np.pi * hour / 24.0),
        'dow_sin': np.sin(2 * np.pi * dow / 7.0),
        'dow_cos': np.cos(2 * np.pi * dow / 7.0),
        'is_weekend': int(dow >= 5),
        'lag_1_step': lag_1, 'lag_2_step': lag_2, 'lag_4_step': lag_4, 'lag_48_step': lag_48,
        'roll_mean_4': roll_mean_4, 'roll_std_4': roll_std_4, 'zone_hist_mean': zone_hist_mean
    }

def predict_all_zones(target_time):
    model, history_df, centroids_df = load_resources()
    rows = []
    for zone_id in centroids_df['zone_dbscan']:
        feats = _engineer_features_for_target(zone_id, target_time, _zone_series_cache)
        if feats is None:
            continue
        X = pd.DataFrame([feats])[FEATURES]
        pred = max(0, model.predict(X)[0])
        rows.append({'zone_dbscan': str(zone_id), 'predicted_count': pred})

    if not rows:
        raise ValueError("No historical data exists before this timestamp for any zone — pick a later date/time.")

    pred_df = pd.DataFrame(rows).merge(centroids_df, on='zone_dbscan', how='left')
    return pred_df.sort_values('predicted_count', ascending=False)

def get_actual_for_slot(target_time):
    """For the compare page — pulls real counts for a slot that exists in history (test period)."""
    _, history_df, centroids_df = load_resources()
    actual_slice = history_df[history_df['created_datetime'] == target_time]
    merged = actual_slice.merge(centroids_df, on='zone_dbscan', how='left')
    return merged.sort_values('violation_count', ascending=False)